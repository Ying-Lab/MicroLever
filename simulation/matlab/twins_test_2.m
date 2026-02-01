%% find_dynamical_twins_v25_full_package.m
% 终极生产版 V25：全套数据输出 (H5 + PDF + CSV)
% 1. 核心逻辑保持 V24 (Time=0:1:30, 修复 getLabel)
% 2. 输出完全适配训练流程的 CSV 报表 (csv_reports 文件夹)
clear; clc;

%% ========== 1. 生产参数设置 ==========
targetPairs = 50;  
startCount  = 0;    
rootOutDir = 'dynamical_twins_t15_full'; % 输出目录
if ~exist(rootOutDir,'dir'), mkdir(rootOutDir); end

% ---- 物理模型参数 ----
CDIFF_IDX = 12; HEALTHY_THRESH = 1e-5; DISEASE_THRESH = 0.5;
DONOR_NZ_MAX = 20; RECIP_NZ_MAX = 43; zero_eps = 1e-9;
global MAX_ABUND_THRESH; MAX_ABUND_THRESH = 5;
N = 53; time = 0:1:30;   % 保持原始时间点

% ---- 筛选阈值 ----
bc_lower = 0.05; bc_upper = 0.15; perturb_strength = 0.08;
eig_gap_thresh = 0.2; ftle_gap_thresh = 0.2; slope_mag_thresh = 1e-5;
maxBPerA = 5000; maxDonorTrials = 5000;
MasterSeed = 2025; rng(MasterSeed, 'twister');

% ---- 加载参数 ----
if ~exist('fake_A_53_dHOMO.csv', 'file'), error('缺少文件'); end
A = readmatrix('fake_A_53_dHOMO.csv');
diag_indices = 1:size(A,1)+1:numel(A);
if any(A(diag_indices) > -0.05), A(diag_indices(A(diag_indices) > -0.05)) = -0.5; end

r_global = [0.546393682769112 0.760433327590399 0.0353683323691972 0.974659046968858 ...
     0.731807335391887 0.0710462328454989 0.613604090023761 0.834371235488301 ...
     0.126546903200044 0.351623621919953 0.576280255262565 0.790223729862463 ...
     0.891626888086389 0.772899504539558 0.0986095500572747 0.420563315102411 ...
     0.0506158387011274 0.483520458174648 0.672017485748202 0.0934335632165327 ...
     0.817447862948387 0.128099409461152 0.207226049121578 0.976336430477113 ...
     0.0584022140987492 0.838636627039298 0.0948175681803862 0.119278286710997 ...
     0.0491398723843669 0.981760843813893 0.226782961438554 0.468696080563501 ...
     0.869907586826040 0.631771781423754 0.421126267492159 0.890388402891514 ...
     0.203172252659994 0.991435713250166 0.916212036673806 0.913836269628040 ...
     0.333467410006688 0.378921990452184 0.911069611511989 0.0141651713399514 ...
     0.546293708505807 0.425619411235411 0.313143856305284 0.858090795614043 ...
     0.848109146766575 0.0729100072579711 0.961585416834993 0.114327988406995 ...
     0.194123580993973];
r_global = r_global(:)'; 

%% ========== HDF5 初始化 ==========
h5file = fullfile(rootOutDir, 'Dynamical_Twins_Data.h5');
if exist(h5file,'file'), delete(h5file); end
h5create(h5file,'/time',[length(time) 1]); h5write(h5file,'/time',time');
h5create(h5file,'/metadata/species_index',[N 1],'Datatype','int32');
h5write(h5file,'/metadata/species_index',int32((1:N)'));
h5writeatt(h5file,'/metadata/species_index','Cdiff_index',CDIFF_IDX);

% --- CSV 数据缓存容器 ---
% 使用 Map 存储向量数据，Key为ID (String)，Value为向量
donor_data_map = containers.Map(); 
recip_data_map = containers.Map();
recip_r_map    = containers.Map();
% 使用 Struct 数组存储稀疏矩阵数据 (Interaction)
interaction_records = struct('rid', {}, 'did', {}, 'label', {}, 'val_real', {});

all_recipient_ids = {}; 
all_donor_ids_list = {}; 

%% ========== 2. 生产主循环 ==========
pairCount = startCount;
global_trial_counter = 0; 
fprintf('Starting V25 (Full Suite): Target = %d Pairs\n', targetPairs);

while pairCount < targetPairs
    global_trial_counter = global_trial_counter + 1;
    seedA = uint32(MasterSeed + global_trial_counter * 10000); 
    
    if mod(global_trial_counter, 50) == 0
        fprintf('... Scanning (Trials: %d, Found: %d/%d) ...\n', ...
            global_trial_counter, pairCount, targetPairs);
    end
    
    % --- 生成受体 A ---
    recipA = genHeteroRecipient(A, r_global, perturb_strength, time, CDIFF_IDX, zero_eps, RECIP_NZ_MAX, DISEASE_THRESH, seedA);
    if isempty(recipA), continue; end
    maskA = recipA.mask; tsA_rel = normalizeCols(recipA.ts); 
    similarFound = false; bestCandidate = struct(); check_fail_count = 0; 
    
    %% 内层1：找相似的 B
    for bTrial = 1:maxBPerA
        if check_fail_count > 100, break; end 
        seedB = uint32(seedA + bTrial); 
        recipB = genHeteroRecipient_FixedMask(A, r_global, perturb_strength, time, CDIFF_IDX, zero_eps, RECIP_NZ_MAX, DISEASE_THRESH, seedB, maskA);
        if isempty(recipB), continue; end
        
        [minDist, tIdxA, tIdxB] = findMinSnapshotDist_preA(tsA_rel, recipA.ts, recipB.ts);
        if minDist > 0.3, check_fail_count = check_fail_count + 1; continue; end
        
        if minDist >= bc_lower && minDist <= bc_upper
            check_fail_count = 0; 
            snapshotA = recipA.ts(:, tIdxA); snapshotB = recipB.ts(:, tIdxB);
            
            slopeA = getSpecificSlope(snapshotA, recipA.r_local, A, CDIFF_IDX);
            slopeB = getSpecificSlope(snapshotB, recipB.r_local, A, CDIFF_IDX);
            if (slopeA * slopeB >= 0) || (abs(slopeA) < slope_mag_thresh) || (abs(slopeB) < slope_mag_thresh), continue; end
            
            eigA = getMaxEigenvalue(snapshotA, recipA.r_local, A);
            eigB = getMaxEigenvalue(snapshotB, recipB.r_local, A);
            if abs(eigA - eigB) < eig_gap_thresh, continue; end
            
            tau = 1.0; 
            ftleA = getFTLE_numerical(snapshotA, recipA.r_local, A, tau);
            ftleB = getFTLE_numerical(snapshotB, recipB.r_local, A, tau);
            if abs(ftleA - ftleB) < ftle_gap_thresh, continue; end
            
            similarFound = true;
            bestCandidate = struct('recipB',recipB, 'minDist',minDist, 'tIdxA',tIdxA, 'tIdxB',tIdxB, ...
                                   'eigA',eigA, 'eigB',eigB, 'ftleA',ftleA, 'ftleB',ftleB, ...
                                   'slopeA',slopeA, 'slopeB',slopeB, 'snapshotA', snapshotA, 'snapshotB', snapshotB);
            break; 
        end
    end
    if ~similarFound, continue; end
    
    recipB = bestCandidate.recipB;
    tIdxA = bestCandidate.tIdxA; tIdxB = bestCandidate.tIdxB;
    snapshotA = bestCandidate.snapshotA; snapshotB = bestCandidate.snapshotB;
    
    % 自然病程检查 (Safe Check)
    [~, endA_nat] = safe_runGLV(snapshotA, A, recipA.r_local, time);
    if isempty(endA_nat), continue; end
    [~, endB_nat] = safe_runGLV(snapshotB, A, recipB.r_local, time);
    if isempty(endB_nat), continue; end
    
    if getLabel(endA_nat, CDIFF_IDX, HEALTHY_THRESH, DISEASE_THRESH) == 1 || ...
       getLabel(endB_nat, CDIFF_IDX, HEALTHY_THRESH, DISEASE_THRESH) == 1
       continue; 
    end

    %% 内层2：测试供体
    for dTrial = 1:maxDonorTrials
        DONOR_OFFSET = 5000000;
        seedD = uint32(seedA + DONOR_OFFSET + dTrial);
        [donorFinal, okD] = genHealthyDonor_singleSeed(A, r_global, time, CDIFF_IDX, zero_eps, DONOR_NZ_MAX, HEALTHY_THRESH, seedD);
        if ~okD, continue; end
        
        % Instant
        [XX_A_imm, xA_end_imm] = safe_runGLV(snapshotA + donorFinal, A, recipA.r_local, time);
        if isempty(XX_A_imm), continue; end
        [XX_B_imm, xB_end_imm] = safe_runGLV(snapshotB + donorFinal, A, recipB.r_local, time);
        if isempty(XX_B_imm), continue; end
        
        labA_imm = getLabel(xA_end_imm, CDIFF_IDX, HEALTHY_THRESH, DISEASE_THRESH);
        labB_imm = getLabel(xB_end_imm, CDIFF_IDX, HEALTHY_THRESH, DISEASE_THRESH);
        
        if isnan(labA_imm) || isnan(labB_imm) || (labA_imm == labB_imm), continue; end
        
        % Delayed
        xA_30 = recipA.ts(:, end); [XX_A_del, xA_end_del] = safe_runGLV(xA_30 + donorFinal, A, recipA.r_local, time);
        if isempty(XX_A_del), continue; end
        xB_30 = recipB.ts(:, end); [XX_B_del, xB_end_del] = safe_runGLV(xB_30 + donorFinal, A, recipB.r_local, time);
        if isempty(XX_B_del), continue; end
        labA_del = getLabel(xA_end_del, CDIFF_IDX, HEALTHY_THRESH, DISEASE_THRESH);
        labB_del = getLabel(xB_end_del, CDIFF_IDX, HEALTHY_THRESH, DISEASE_THRESH);
        
        if isnan(labA_del) || (labA_imm ~= labA_del), continue; end
        if isnan(labB_del) || (labB_imm ~= labB_del), continue; end
        
        % === Success ===
        pairCount = pairCount + 1;
        pairInfo = bestCandidate;
        pairInfo.recipientA_id = recipA.id; pairInfo.recipientA_seed = recipA.seed;
        pairInfo.recipientB_id = recipB.id; pairInfo.recipientB_seed = recipB.seed;
        pairInfo.rA_local = recipA.r_local; pairInfo.rB_local = recipB.r_local;
        pairInfo.timeA = time(tIdxA); pairInfo.timeB = time(tIdxB);
        pairInfo.donor_id = sprintf('D_%u', seedD); pairInfo.donor_seed = seedD;
        pairInfo.donorFinal = donorFinal;
        pairInfo.outcomeA_imm = labA_imm; pairInfo.outcomeA_del = labA_del;
        pairInfo.outcomeB_imm = labB_imm; pairInfo.outcomeB_del = labB_del;
        pairInfo.tsA = recipA.ts; pairInfo.tsB = recipB.ts;
        pairInfo.tsA_fmt = XX_A_imm'; pairInfo.tsB_fmt = XX_B_imm';
        pairInfo.timeGrid = time; pairInfo.CDIFF_IDX = CDIFF_IDX;
        
        fprintf('\n>>> [Success] Pair %d / %d Found! <<<\n', pairCount, targetPairs);
        
        % 保存可视化
        pairDir = fullfile(rootOutDir, sprintf('pair_%03d', pairCount));
        if ~exist(pairDir,'dir'), mkdir(pairDir); end
        save(fullfile(pairDir,'pairInfo.mat'),'pairInfo');
        plot_comparison_abs(pairInfo, A, fullfile(pairDir, 'Comparison_Absolute.png')); 
        plot_comparison_rel(pairInfo, A, fullfile(pairDir, 'Comparison_Relative.png'));
        
        % === HDF5 & CSV 数据收集 ===
        rid_A = sprintf('Pair_%03d_A', pairCount);
        rid_B = sprintf('Pair_%03d_B', pairCount);
        did   = sprintf('D%06d', seedD);
        
        % 1. HDF5 写入
        dset_Ra = ['/recipients/' rid_A '/data']; delete_h5_path(h5file, dset_Ra);
        h5create(h5file, dset_Ra, size(recipA.ts')); h5write(h5file, dset_Ra, recipA.ts');
        dset_Ra_r = ['/recipients/' rid_A '/r']; delete_h5_path(h5file, dset_Ra_r);
        h5create(h5file, dset_Ra_r, [N 1]); h5write(h5file, dset_Ra_r, recipA.r_local.');
        
        dset_Rb = ['/recipients/' rid_B '/data']; delete_h5_path(h5file, dset_Rb);
        h5create(h5file, dset_Rb, size(recipB.ts')); h5write(h5file, dset_Rb, recipB.ts');
        dset_Rb_r = ['/recipients/' rid_B '/r']; delete_h5_path(h5file, dset_Rb_r);
        h5create(h5file, dset_Rb_r, [N 1]); h5write(h5file, dset_Rb_r, recipB.r_local.');
        
        tpath_A = sprintf('/transplants/%s/%s/data', did, rid_A); delete_h5_path(h5file, tpath_A);
        h5create(h5file, tpath_A, size(XX_A_imm)); h5write(h5file, tpath_A, XX_A_imm);
        tpath_B = sprintf('/transplants/%s/%s/data', did, rid_B); delete_h5_path(h5file, tpath_B);
        h5create(h5file, tpath_B, size(XX_B_imm)); h5write(h5file, tpath_B, XX_B_imm);
        
        % 2. CSV 数据缓存
        if ~isKey(donor_data_map, did), donor_data_map(did) = donorFinal; end
        
        recip_data_map(rid_A) = recipA.ts(:, end); % 记录受体病态终点 (Pre-FMT)
        recip_data_map(rid_B) = recipB.ts(:, end);
        recip_r_map(rid_A)    = recipA.r_local;
        recip_r_map(rid_B)    = recipB.r_local;
        
        % 记录互作关系 (Label & Real Value)
        val_real_A = xA_end_imm(CDIFF_IDX, end);
        val_real_B = xB_end_imm(CDIFF_IDX, end);
        
        interaction_records(end+1) = struct('rid', rid_A, 'did', did, 'label', labA_imm, 'val_real', val_real_A);
        interaction_records(end+1) = struct('rid', rid_B, 'did', did, 'label', labB_imm, 'val_real', val_real_B);
        
        all_recipient_ids = [all_recipient_ids; {rid_A}; {rid_B}];
        all_donor_ids_list = [all_donor_ids_list; {did}];
        
        close all; break; 
    end
end

%% ========== 3. 生成 CSV 报表 & HDF5 汇总标签 ==========
fprintf('\nGenerating CSV Reports & Final HDF5 Labels...\n');

csv_dir = fullfile(rootOutDir, 'csv_reports');
if ~exist(csv_dir, 'dir'), mkdir(csv_dir); end

unique_donor_ids = unique(all_donor_ids_list, 'stable'); 
all_recipient_ids = all_recipient_ids(:); 
num_donors = length(unique_donor_ids); 
num_recips = length(all_recipient_ids);

% 构建矩阵
C_diff_labels = NaN(num_recips, num_donors);
C_diff_real   = NaN(num_recips, num_donors);
donor_matrix  = zeros(N, num_donors);
recip_matrix  = zeros(N, num_recips);
recip_r_matrix= zeros(N, num_recips);

% 填充 Donor 矩阵
for i = 1:num_donors
    donor_matrix(:, i) = donor_data_map(unique_donor_ids{i});
end

% 填充 Recipient 矩阵
for i = 1:num_recips
    recip_matrix(:, i) = recip_data_map(all_recipient_ids{i});
    recip_r_matrix(:, i) = recip_r_map(all_recipient_ids{i});
end

% 填充 Interaction 矩阵 (Sparse)
donor_map = containers.Map(unique_donor_ids, 1:num_donors);
recip_map = containers.Map(all_recipient_ids, 1:num_recips);

for k = 1:length(interaction_records)
    r_key = interaction_records(k).rid;
    d_key = interaction_records(k).did;
    r_idx = recip_map(r_key);
    d_idx = donor_map(d_key);
    
    C_diff_labels(r_idx, d_idx) = interaction_records(k).label;
    C_diff_real(r_idx, d_idx)   = interaction_records(k).val_real;
end

% --- 写入 CSV 文件 ---
species_hdr = arrayfun(@(x)sprintf('S%02d',x), 1:N, 'UniformOutput', false);

% 1. Donor Final Abundance
writetable(array2table(donor_matrix, 'RowNames', species_hdr, 'VariableNames', unique_donor_ids), ...
    fullfile(csv_dir, 'donor_final_abundance.csv'), 'WriteRowNames', true);

% 2. Recipient Final Abundance
writetable(array2table(recip_matrix, 'RowNames', species_hdr, 'VariableNames', all_recipient_ids), ...
    fullfile(csv_dir, 'recipient_final_abundance.csv'), 'WriteRowNames', true);

% 3. Transplant C.diff Real
writetable(array2table(C_diff_real, 'RowNames', all_recipient_ids, 'VariableNames', unique_donor_ids), ...
    fullfile(csv_dir, 'transplant_cdiff_real.csv'), 'WriteRowNames', true);

% 4. Recipient Growth Rates
writetable(array2table(recip_r_matrix, 'RowNames', species_hdr, 'VariableNames', all_recipient_ids), ...
    fullfile(csv_dir, 'recipient_growth_rates.csv'), 'WriteRowNames', true);

% 5. C.diff Labels (Matrix & Table)
writematrix(C_diff_labels, fullfile(csv_dir, 'Cdiff_labels.csv'));
writetable(array2table(C_diff_labels, 'RowNames', all_recipient_ids, 'VariableNames', unique_donor_ids), ...
    fullfile(csv_dir, 'Cdiff_labels_table.csv'), 'WriteRowNames', true);

% --- 写入 HDF5 剩余部分 ---
h5create(h5file, '/transplants/donor_ids', [num_donors 1], 'Datatype', 'string');
h5write(h5file, '/transplants/donor_ids', unique_donor_ids);
h5create(h5file, '/transplants/recipient_ids', [num_recips 1], 'Datatype', 'string');
h5write(h5file, '/transplants/recipient_ids', all_recipient_ids);
h5create(h5file, '/transplants/C_diff_labels', size(C_diff_labels));
h5write(h5file, '/transplants/C_diff_labels', C_diff_labels);

fprintf('Done. All files (H5, PDF, CSV) generated in: %s\n', rootOutDir);

%% ==================== 核心函数 ====================
function delete_h5_path(h5file, path)
    fid = H5F.open(h5file, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    try H5L.delete(fid, path, 'H5P_DEFAULT'); catch, end
    H5F.close(fid);
end
function ftle = getFTLE_numerical(x0, r, A, tau)
    epsilon = 1e-6; norm_x0 = norm(x0); if norm_x0 < 1e-9, ftle = -10; return; end
    delta_vec = epsilon * (x0 ./ norm_x0); x0_pert = x0 + delta_vec; t_short = 0:0.1:tau; 
    [~, x_end] = safe_runGLV_short(x0, A, r, t_short);
    if isempty(x_end), ftle = NaN; return; end
    [~, x_pert_end] = safe_runGLV_short(x0_pert, A, r, t_short);
    if isempty(x_pert_end), ftle = NaN; return; end
    final_dist = norm(x_end - x_pert_end); init_dist = norm(delta_vec);
    if final_dist < 1e-12, final_dist = 1e-12; end 
    ftle = (1 / abs(tau)) * log(final_dist / init_dist);
end
function [ts, x_end] = safe_runGLV_short(x0, A, r, time_grid)
    global MAX_ABUND_THRESH; N = numel(x0); nt = numel(time_grid); dt = time_grid(2)-time_grid(1);
    dxx = x0(:); r = r(:); deriv_func = @(x) x .* (A' * x + r);
    for i = 2:nt
        k1 = dt * deriv_func(dxx); xx=dxx+0.5*k1; xx(xx<0)=0; k2=dt*deriv_func(xx); xx=dxx+0.5*k2; xx(xx<0)=0;
        k3 = dt * deriv_func(xx);  xx=dxx+k3;     xx(xx<0)=0; k4=dt*deriv_func(xx);
        dxx = dxx + (k1+2*k2+2*k3+k4)/6; dxx(dxx<0)=0;
        if max(dxx) > MAX_ABUND_THRESH, ts=[]; x_end=[]; return; end
    end
    ts = 1; x_end = dxx;
end
%% ==================== 绘图函数 ====================
function plot_comparison_abs(pairInfo, A, outPath)
    CDIFF_IDX = pairInfo.CDIFF_IDX; N = 53; MyColor = lines(N); 
    xA_final_delayed = pairInfo.tsA(:, end); 
    [ts_FMT_A_delayed_raw, ~] = safe_runGLV(xA_final_delayed + pairInfo.donorFinal, A, pairInfo.rA_local, pairInfo.timeGrid);
    if isempty(ts_FMT_A_delayed_raw), tsA_fmt_delayed = NaN(N, length(pairInfo.timeGrid)); else, tsA_fmt_delayed = ts_FMT_A_delayed_raw'; end
    xB_final_delayed = pairInfo.tsB(:, end); 
    [ts_FMT_B_delayed_raw, ~] = safe_runGLV(xB_final_delayed + pairInfo.donorFinal, A, pairInfo.rB_local, pairInfo.timeGrid);
    if isempty(ts_FMT_B_delayed_raw), tsB_fmt_delayed = NaN(N, length(pairInfo.timeGrid)); else, tsB_fmt_delayed = ts_FMT_B_delayed_raw'; end
    t_A_pre = pairInfo.timeGrid(1:pairInfo.tIdxA); t_shift_A = pairInfo.timeGrid(pairInfo.tIdxA); t_all_A_imm = [t_A_pre, t_shift_A + pairInfo.timeGrid]; ALL_A_imm = [pairInfo.tsA(:, 1:pairInfo.tIdxA), pairInfo.tsA_fmt];
    t_B_pre = pairInfo.timeGrid(1:pairInfo.tIdxB); t_shift_B = pairInfo.timeGrid(pairInfo.tIdxB); t_all_B_imm = [t_B_pre, t_shift_B + pairInfo.timeGrid]; ALL_B_imm = [pairInfo.tsB(:, 1:pairInfo.tIdxB), pairInfo.tsB_fmt];
    t_del = [pairInfo.timeGrid, pairInfo.timeGrid(end) + pairInfo.timeGrid]; ALL_A_del = [pairInfo.tsA, tsA_fmt_delayed]; ALL_B_del = [pairInfo.tsB, tsB_fmt_delayed];
    
    fig = figure('Position', [50, 50, 1000, 800], 'Visible', 'off'); t = tiledlayout(2, 2, 'Padding', 'compact');
    nexttile; hold on; for i=1:N, plot(t_all_A_imm, ALL_A_imm(i,:), 'Color', MyColor(i,:), 'LineWidth', 0.5); end; plot(t_all_A_imm, ALL_A_imm(CDIFF_IDX,:), 'r', 'LineWidth', 2.5);
    xline(pairInfo.timeA, 'k--', {sprintf('t=%.1f', pairInfo.timeA), sprintf('FTLE=%.3f', pairInfo.ftleA), sprintf('Eig=%.2f', pairInfo.eigA), sprintf('Slope=%.1e', pairInfo.slopeA)}, 'FontSize', 8, 'LabelVerticalAlignment','bottom', 'LabelHorizontalAlignment','right');
    title(sprintf('A (Seed %d) Instant -> %d', pairInfo.recipientA_seed, pairInfo.outcomeA_imm)); xlim([0, t_all_A_imm(end)]); ylabel('Abs Abundance'); grid on;
    nexttile; hold on; for i=1:N, plot(t_all_B_imm, ALL_B_imm(i,:), 'Color', MyColor(i,:), 'LineWidth', 0.5); end; plot(t_all_B_imm, ALL_B_imm(CDIFF_IDX,:), 'r', 'LineWidth', 2.5);
    xline(pairInfo.timeB, 'k--', {sprintf('t=%.1f', pairInfo.timeB), sprintf('FTLE=%.3f', pairInfo.ftleB), sprintf('Eig=%.2f', pairInfo.eigB), sprintf('Slope=%.1e', pairInfo.slopeB)}, 'FontSize', 8, 'LabelVerticalAlignment','bottom', 'LabelHorizontalAlignment','right');
    title(sprintf('B (Seed %d) Instant -> %d', pairInfo.recipientB_seed, pairInfo.outcomeB_imm)); xlim([0, t_all_B_imm(end)]); grid on;
    nexttile; hold on; for i=1:N, plot(t_del, ALL_A_del(i,:), 'Color', MyColor(i,:), 'LineWidth', 0.5); end; plot(t_del, ALL_A_del(CDIFF_IDX,:), 'r', 'LineWidth', 2.5);
    xline(30, 'k--', 'Delayed', 'LabelVerticalAlignment','bottom'); title(sprintf('A Delayed -> %d', pairInfo.outcomeA_del)); xlim([0, t_del(end)]); ylabel('Abs Abundance'); xlabel('Time'); grid on;
    nexttile; hold on; for i=1:N, plot(t_del, ALL_B_del(i,:), 'Color', MyColor(i,:), 'LineWidth', 0.5); end; plot(t_del, ALL_B_del(CDIFF_IDX,:), 'r', 'LineWidth', 2.5);
    xline(30, 'k--', 'Delayed', 'LabelVerticalAlignment','bottom'); title(sprintf('B Delayed -> %d', pairInfo.outcomeB_del)); xlim([0, t_del(end)]); xlabel('Time'); grid on;
    title(t, sprintf('Dynamical Twins: BC=%.4f | Donor Seed %d', pairInfo.minDist, pairInfo.donor_seed), 'FontSize', 14); 
    
    exportgraphics(fig, outPath, 'Resolution', 300);
    [p, n, ~] = fileparts(outPath);
    exportgraphics(fig, fullfile(p, [n, '.pdf']), 'ContentType', 'vector');
end
function plot_comparison_rel(pairInfo, A, outPath)
    CDIFF_IDX = pairInfo.CDIFF_IDX; N = 53; MyColor = lines(N);
    xA_final_delayed = pairInfo.tsA(:, end); 
    [ts_FMT_A_delayed_raw, ~] = safe_runGLV(xA_final_delayed + pairInfo.donorFinal, A, pairInfo.rA_local, pairInfo.timeGrid);
    if isempty(ts_FMT_A_delayed_raw), tsA_fmt_delayed = NaN(N, length(pairInfo.timeGrid)); else, tsA_fmt_delayed = ts_FMT_A_delayed_raw'; end
    xB_final_delayed = pairInfo.tsB(:, end); 
    [ts_FMT_B_delayed_raw, ~] = safe_runGLV(xB_final_delayed + pairInfo.donorFinal, A, pairInfo.rB_local, pairInfo.timeGrid);
    if isempty(ts_FMT_B_delayed_raw), tsB_fmt_delayed = NaN(N, length(pairInfo.timeGrid)); else, tsB_fmt_delayed = ts_FMT_B_delayed_raw'; end
    t_A_pre = pairInfo.timeGrid(1:pairInfo.tIdxA); t_shift_A = pairInfo.timeGrid(pairInfo.tIdxA); t_all_A_imm = [t_A_pre, t_shift_A + pairInfo.timeGrid]; ALL_A_imm_rel = normalizeCols([pairInfo.tsA(:, 1:pairInfo.tIdxA), pairInfo.tsA_fmt]);
    t_B_pre = pairInfo.timeGrid(1:pairInfo.tIdxB); t_shift_B = pairInfo.timeGrid(pairInfo.tIdxB); t_all_B_imm = [t_B_pre, t_shift_B + pairInfo.timeGrid]; ALL_B_imm_rel = normalizeCols([pairInfo.tsB(:, 1:pairInfo.tIdxB), pairInfo.tsB_fmt]);
    t_del = [pairInfo.timeGrid, pairInfo.timeGrid(end) + pairInfo.timeGrid]; ALL_A_del_rel = normalizeCols([pairInfo.tsA, tsA_fmt_delayed]); ALL_B_del_rel = normalizeCols([pairInfo.tsB, tsB_fmt_delayed]);
    
    fig = figure('Position', [50, 50, 1000, 800], 'Visible', 'off'); t = tiledlayout(2, 2, 'Padding', 'compact');
    nexttile; hold on; for i=1:N, plot(t_all_A_imm, ALL_A_imm_rel(i,:), 'Color', MyColor(i,:)); end; plot(t_all_A_imm, ALL_A_imm_rel(CDIFF_IDX,:), 'r', 'LineWidth', 2.5);
    xline(pairInfo.timeA, 'k--', {sprintf('t=%.1f', pairInfo.timeA), sprintf('FTLE=%.3f', pairInfo.ftleA)}, 'FontSize', 8, 'LabelVerticalAlignment','bottom'); title(sprintf('A Instant (Seed %d)', pairInfo.recipientA_seed)); ylim([0 1]); ylabel('Rel Abundance');
    nexttile; hold on; for i=1:N, plot(t_all_B_imm, ALL_B_imm_rel(i,:), 'Color', MyColor(i,:)); end; plot(t_all_B_imm, ALL_B_imm_rel(CDIFF_IDX,:), 'r', 'LineWidth', 2.5);
    xline(pairInfo.timeB, 'k--', {sprintf('t=%.1f', pairInfo.timeB), sprintf('FTLE=%.3f', pairInfo.ftleB)}, 'FontSize', 8, 'LabelVerticalAlignment','bottom'); title(sprintf('B Instant (Seed %d)', pairInfo.recipientB_seed)); ylim([0 1]);
    nexttile; hold on; for i=1:N, plot(t_del, ALL_A_del_rel(i,:), 'Color', MyColor(i,:)); end; plot(t_del, ALL_A_del_rel(CDIFF_IDX,:), 'r', 'LineWidth', 2.5);
    xline(30, 'k--', 'Delayed'); title('A Delayed'); xlabel('Time'); ylabel('Rel Abundance'); ylim([0 1]);
    nexttile; hold on; for i=1:N, plot(t_del, ALL_B_del_rel(i,:), 'Color', MyColor(i,:)); end; plot(t_del, ALL_B_del_rel(CDIFF_IDX,:), 'r', 'LineWidth', 2.5);
    xline(30, 'k--', 'Delayed'); title('B Delayed'); xlabel('Time'); ylim([0 1]);
    title(t, 'Relative Dynamics', 'FontSize', 14); 
    
    exportgraphics(fig, outPath, 'Resolution', 300);
    [p, n, ~] = fileparts(outPath);
    exportgraphics(fig, fullfile(p, [n, '.pdf']), 'ContentType', 'vector');
end
% --- 生成与计算 ---
function slope = getSpecificSlope(x, r, A, target_idx), r=r(:); x=x(:); dx=x.*(A'*x+r); slope=dx(target_idx); end
function lab = getLabel(x, idx, h, d), val=x(idx); if val<h, lab=1; elseif val>d, lab=0; else, lab=NaN; end, end
function recip = genHeteroRecipient(A, r_global, p_str, time, CDIFF_IDX, zero_eps, MAX_NZ, DIS_THR, seed), sc=rng; rng(seed,'twister'); N=numel(r_global); 
    for k=1:100, r_local=r_global.*(1+p_str*randn(1,N)); x0_raw=rand(N,1); num_del=randi([35,40]); del=setdiff(randsample(N,num_del),CDIFF_IDX); x0=x0_raw; x0(del)=0; x0(CDIFF_IDX)=0.01; mask=(x0>0); mask(CDIFF_IDX)=true; [XX,endv]=safe_runGLV(x0,A,r_local,time); if ~isempty(XX)&&endv(CDIFF_IDX)>=DIS_THR&&sum(endv<zero_eps,'all')<=MAX_NZ, recip.id=sprintf('R_%u',seed); recip.seed=seed; recip.r_local=r_local; recip.mask=mask; recip.ts=XX'; recip.final=endv; rng(sc); return; end, end, recip=[]; rng(sc); end
function recip = genHeteroRecipient_FixedMask(A, r_global, p_str, time, CDIFF_IDX, zero_eps, MAX_NZ, DIS_THR, seed, maskA), sc=rng; rng(seed,'twister'); N=numel(r_global);
    for k=1:100, r_local=r_global.*(1+p_str*randn(1,N)); x0=zeros(N,1); idx=find(maskA); x0(idx)=rand(numel(idx),1); x0(CDIFF_IDX)=0.01; [XX,endv]=safe_runGLV(x0,A,r_local,time); if ~isempty(XX)&&endv(CDIFF_IDX)>=DIS_THR&&sum(endv<zero_eps,'all')<=MAX_NZ, recip.id=sprintf('R_%u',seed); recip.seed=seed; recip.r_local=r_local; recip.mask=maskA; recip.ts=XX'; recip.final=endv; rng(sc); return; end, end, recip=[]; rng(sc); end
function [d_end, ok] = genHealthyDonor_singleSeed(A, r, time, CDIFF_IDX, zero_eps, DONOR_NZ_MAX, HEALTHY_THRESH, seed), sc=rng; rng(seed,'twister'); N=numel(r); x0=rand(N,1); k=round(rand*0.3*N); if k>0, x0(randperm(N,k))=0; end; x0(CDIFF_IDX)=(rand<0.5)*0.01; [XX,d_end]=safe_runGLV(x0,A,r,time); rng(sc); if isempty(XX)||sum(d_end<zero_eps,'all')>DONOR_NZ_MAX||d_end(CDIFF_IDX)>=HEALTHY_THRESH, ok=false; return; end, ok=true; end
function lambda = getMaxEigenvalue(x, r, A), N=length(x); J=(A'.*x')'; inter=A'*x; g=r(:)+inter; for i=1:N, J(i,i)=g(i)+x(i)*A(i,i); end, lambda=max(real(eig(J))); end
function [minDist, tIdxA, tIdxB] = findMinSnapshotDist_preA(tsA, tsA_raw, tsB_raw), [~,TA]=size(tsA); [~,TB]=size(tsB_raw); tsB=normalizeCols(tsB_raw); minDist=inf; tIdxA=1; tIdxB=1; for a=1:TA, p=tsA(:,a); for b=1:TB, q=tsB(:,b); d=sum(abs(p-q))/2; if d<minDist, minDist=d; tIdxA=a; tIdxB=b; end, end, end, end
function X_rel = normalizeCols(X), s=sum(X,1); s(s==0)=1; X_rel=X./s; end
function [ts, x_end] = safe_runGLV(x0, A, r, time), global MAX_ABUND_THRESH; [XX, x_end] = glv_RK4_type_safe_monitor(x0, A, r, time); if any(isnan(XX(:))) || max(x_end, [], 'all') > MAX_ABUND_THRESH, ts=[]; x_end=[]; else, ts=XX; end, end
function [dx, dxx] = glv_RK4_type_safe_monitor(x, A, r, time), global MAX_ABUND_THRESH; N=size(A,1); nt=numel(time); dt=time(2)-time(1); dxx=x(:); dx=zeros(N,nt); dx(:,1)=dxx; r=r(:); func=@(x) x.*(A'*x+r); for i=2:nt, k1=dt*func(dxx); xk=dxx+0.5*k1; xk(xk<0)=0; k2=dt*func(xk); xk=dxx+0.5*k2; xk(xk<0)=0; k3=dt*func(xk); xk=dxx+k3; xk(xk<0)=0; k4=dt*func(xk); dxx=dxx+(k1+2*k2+2*k3+k4)/6; dxx(dxx<0)=0; if max(dxx)>MAX_ABUND_THRESH, dx=NaN(size(dx)); dxx=NaN(size(dxx)); return; end, dx(:,i)=dxx; end, dx=dx'; end