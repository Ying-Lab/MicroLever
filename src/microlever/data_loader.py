# dataloader.py
import numpy as np
import h5py, torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader

# ------------------------ 1. 伪造工具 ------------------------
def corrupt_edge_weights(edge_weights,
                         corruption_rate: float = 0.0,
                         mode: str = "none",
                         rng: torch.Generator | None = None):
    if corruption_rate <= 0.0 or mode == "none":
        return edge_weights
    E = edge_weights.size(0)
    num_bad = max(1, int(E * corruption_rate))
    rng = rng or torch.Generator(device=edge_weights.device)
    sel = torch.randperm(E, generator=rng, device=edge_weights.device)[:num_bad]
    ew_new = edge_weights.clone()
    if mode in {"sign", "both"}:
        ew_new[sel] = -ew_new[sel]
    if mode in {"value", "both"}:
        factors = torch.empty(num_bad, device=edge_weights.device).uniform_(0.5, 1.5)
        ew_new[sel] = ew_new[sel] * factors
    return ew_new

# ------------------------ 2. CLR / 采样 / 噪声 工具 ------------------------
def robust_clr_transform(matrix, epsilon=1e-8, min_positive=1e-12):
    matrix = np.where(matrix < min_positive, min_positive, matrix)
    matrix = np.where(matrix == 0, epsilon, matrix)
    log_matrix = np.log(matrix)
    geom_mean = np.exp(log_matrix.mean(axis=1, keepdims=True))
    return np.clip(log_matrix - np.log(geom_mean), -50, 50).astype(np.float32)

def ordered_time_sampling(rec_seq, down_sample_step=10,
                          num_samples=5, sampling_mode='random', seed=None):
    downsampled = rec_seq[::down_sample_step]
    if seed is not None:
        np.random.seed(seed)
    if sampling_mode == 'front':
        cands = np.arange(15)
    elif sampling_mode == 'back':
        cands = np.arange(15, 30)
    else:
        cands = np.arange(30)
    idx = np.sort(np.random.choice(cands, num_samples, replace=False))
    return downsampled[idx].astype(np.float32)

def add_nsr_noise(signal, strength, min_signal=1e-12, max_signal=1e6):
    signal = np.clip(signal, min_signal, max_signal)
    log_signal = np.log(np.maximum(signal, min_signal))
    noise_std = np.sqrt(np.exp(2 * log_signal.mean(axis=1, keepdims=True)) * strength)
    noise = np.random.normal(0, noise_std, signal.shape)
    return np.clip(signal + noise, min_signal, max_signal).astype(np.float32)

def add_dm_noise(biomass, theta=0.03, rng=None, depth_mu=10.0, depth_sigma=1.0):
    rng = rng or np.random.default_rng()
    if biomass.ndim == 1:
        biomass = biomass[None, :]
    noisy = np.zeros_like(biomass, dtype=np.float32)
    for i, row in enumerate(biomass):
        total = row.sum()
        if total == 0:
            noisy[i] = row
            continue
        p = row / total
        alpha = p / theta
        L = int(rng.lognormal(depth_mu, depth_sigma))
        counts = rng.dirichlet(alpha) * L
        counts = counts * (total / counts.sum())
        noisy[i] = counts
    return noisy.squeeze().astype(np.float32)

# ------------------------ 3. Dataset（预构建 Data 对象） --------------------------
class FMTDataset:
    def __init__(self, A, rec_seqs, don_states, labels,
                 noise_strength=0.0, classification_threshold=0.01,
                 sampling_mode='random', num_samples=5, data_split='train',
                 sparsity_config=None,
                 corruption_rate=0.0, corruption_mode="none",
                 base_seed=42, use_dm_noise=False,
                 dm_depth_mu=10.0, dm_depth_sigma=1.0):

        self.edge_index = torch.tensor(np.array(np.nonzero(A)).astype(np.int64),
                                       dtype=torch.long)
        self.edge_weight_orig = torch.tensor(
            A[self.edge_index[0], self.edge_index[1]], dtype=torch.float32)

        self.sparsity_cfg   = sparsity_config or {'train': (0.5, 0.9),
                                                  'val':   (0.7, 0.7),
                                                  'test':  (0.7, 0.7)}
        self.data_split     = data_split
        self.noise_strength = noise_strength
        self.use_dm_noise   = use_dm_noise
        self.dm_depth_mu    = dm_depth_mu
        self.dm_depth_sigma = dm_depth_sigma
        self.sampling_mode  = sampling_mode
        self.num_samples    = num_samples
        self.corruption_rate= corruption_rate
        self.corruption_mode= corruption_mode
        self.base_seed      = base_seed

        # 一次性预处理所有样本
        processed = []
        for rec, don, y in zip(rec_seqs, don_states, labels):
            if self.noise_strength > 0:
                if self.use_dm_noise:
                    rec = add_dm_noise(rec, theta=self.noise_strength,
                                       depth_mu=self.dm_depth_mu,
                                       depth_sigma=self.dm_depth_sigma)
                else:
                    rec = add_nsr_noise(rec, self.noise_strength)
            rec /= np.where(rec.sum(1, keepdims=True)==0, 1e-8, rec.sum(1, keepdims=True))
            rec = robust_clr_transform(rec)

            if self.noise_strength > 0:
                if self.use_dm_noise:
                    don = add_dm_noise(don, theta=self.noise_strength,
                                       depth_mu=self.dm_depth_mu,
                                       depth_sigma=self.dm_depth_sigma)
                else:
                    don = add_nsr_noise(don.reshape(1,-1), self.noise_strength).flatten()
            don = robust_clr_transform((don/(don.sum() or 1e-8)).reshape(1,-1)).flatten()
            processed.append((rec, don, float(y)))
            # processed.append((rec, don, float(y < classification_threshold)))

        self.data_list = []
        for idx, (rec, don, label) in enumerate(processed):
            seed = self.base_seed * 100000 + idx

            rec_sampled = ordered_time_sampling(
                rec, num_samples=self.num_samples,
                sampling_mode=self.sampling_mode,
                seed=seed
            )
            node_activity = torch.tensor(rec_sampled.T,
                                         dtype=torch.float32).unsqueeze(-1)
            donor_data = torch.tensor(don, dtype=torch.float32).view(-1,1)

            # 拓扑伪造
            gen = torch.Generator(device='cpu').manual_seed(seed)
            ew_corr = corrupt_edge_weights(
                self.edge_weight_orig,
                corruption_rate=self.corruption_rate,
                mode=self.corruption_mode,
                rng=gen
            )

            # 随机稀疏
            rng = np.random.default_rng(seed)
            min_s, max_s = self.sparsity_cfg[self.data_split]
            keep_ratio = rng.uniform(min_s, max_s)
            E = self.edge_index.size(1)
            keep_num = max(1, int(E * keep_ratio))
            perm = torch.as_tensor(rng.permutation(E)[:keep_num], dtype=torch.long)

            edge_index = self.edge_index[:, perm]
            edge_attr  = ew_corr[perm]

            self.data_list.append(Data(
                x=node_activity[:, -1],
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.float32),
                node_activity=node_activity,
                donor_data=donor_data
            ))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# ------------------------ 4. create_loaders（同步加载） ------------------------
def create_loaders(h5_path, adjacency_path, cdiff_label_path,
                   donor_final_path, receptor_ts_group='/recipients',
                   test_size=0.2, val_size=0.5,
                   noise_strength_train=0.01, noise_strength_val=0.0,
                   noise_strength_test=0.0, batch_size=32, seed=42,
                   sampling_mode='random', num_samples=5,
                   sparsity_config=None,
                   corruption_mode="none",
                   corruption_rates=None,
                   use_dm_noise=False, dm_depth_mu=10.0, dm_depth_sigma=1.0,
                   new_h5_path=None, new_cdiff_label_path=None,
                   new_donor_final_path=None, new_receptor_ts_group=None):

    corruption_rates = corruption_rates or {'train':0.0,'val':0.0,'test':0.0}
    if isinstance(use_dm_noise, bool):
        use_dm_noise = {k:use_dm_noise for k in ('train','val','test')}

    A = pd.read_csv(adjacency_path, header=None).values.astype(np.float32)

    def build(rec, don, lbl, split, noise_strength):
        return FMTDataset(
            A, rec, don, lbl,
            noise_strength=noise_strength,
            sampling_mode=sampling_mode,
            num_samples=num_samples,
            data_split=split,
            sparsity_config=sparsity_config,
            corruption_rate=corruption_rates.get(split,0.0),
            corruption_mode=corruption_mode,
            base_seed=seed,
            use_dm_noise=use_dm_noise.get(split,False),
            dm_depth_mu=dm_depth_mu,
            dm_depth_sigma=dm_depth_sigma
        )

    def load_pairs(h5p, r_group, df_final, cd_label):
        Y = cd_label.values
        pairs = np.where(~np.isnan(Y))
        r_seqs, d_states, lbls = [], [], []
        with h5py.File(h5p,'r') as f:
            g = f[r_group]
            for r,c in zip(*pairs):
                rid = cd_label.index[r]
                seq = g[f'R{int(rid[1:]):03d}/data'][:].astype(np.float32)
                r_seqs.append(seq)
                d_states.append(df_final[:,c].astype(np.float32))
                lbls.append(Y[r,c])
        return r_seqs, d_states, lbls

    donor_final  = pd.read_csv(donor_final_path, index_col=0).values
    cdiff_labels = pd.read_csv(cdiff_label_path, index_col=0)
    rec_seqs, don_states, labels = load_pairs(
        h5_path, receptor_ts_group, donor_final, cdiff_labels)

    if new_h5_path is None:
        tr_r, tmp_r, tr_d, tmp_d, tr_l, tmp_l = train_test_split(
            rec_seqs, don_states, labels, test_size=test_size, random_state=seed)
        va_r, te_r, va_d, te_d, va_l, te_l = train_test_split(
            tmp_r, tmp_d, tmp_l, test_size=val_size, random_state=seed)

        train_set = build(tr_r, tr_d, tr_l, 'train', noise_strength_train)
        val_set   = build(va_r, va_d, va_l, 'val',   noise_strength_val)
        test_set  = build(te_r, te_d, te_l, 'test',  noise_strength_test)
    else:
        tr_r, va_r, tr_d, va_d, tr_l, va_l = train_test_split(
            rec_seqs, don_states, labels, test_size=test_size, random_state=seed)
        donor_final_new  = pd.read_csv(new_donor_final_path, index_col=0).values
        cdiff_labels_new = pd.read_csv(new_cdiff_label_path, index_col=0)
        te_r, te_d, te_l = load_pairs(
            new_h5_path, new_receptor_ts_group,
            donor_final_new, cdiff_labels_new)

        train_set = build(tr_r, tr_d, tr_l, 'train', noise_strength_train)
        val_set   = build(va_r, va_d, va_l, 'val',   noise_strength_val)
        test_set  = build(te_r, te_d, te_l, 'test',  noise_strength_test)

    loader_args = dict(
        num_workers=0,         # 同步加载
        pin_memory=True,
        persistent_workers=False
    )

    return (GeoDataLoader(train_set, batch_size, shuffle=True,  **loader_args),
            GeoDataLoader(val_set,   batch_size, shuffle=False, **loader_args),
            GeoDataLoader(test_set,  batch_size, shuffle=False, **loader_args))
