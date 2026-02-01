%% ==================== FMT 仿真主程序（全局固定 r，增强版）====================
% 修订日期：2025‑05‑02  (随机稀疏 + 终态阈值 + BugFix 全量版)
% -------------------------------------------------------------------------
% 变更摘要
%   1) 供体稀疏化比例 U(0,30%)
%   2) 终态爆炸筛查（> MAX_ABUND_THRESH）
%   3) 全程 NaN 拦截
%   4) 修正 global 变量作用域 & 线程池兼容
% -------------------------------------------------------------------------
% *** 适用于 thread‑based parpool；若改用 process‑based，需 pctRunOnAll ***
% -------------------------------------------------------------------------

%% 1. 运行模式 & 环境清理 ---------------------------------------------------
isTest   = false;                        % false=训练集  true=测试集
trainRange = [1,    5e6];
testRange  = [5e6+1,1e7];
seedRange  = isTest * testRange + (~isTest) * trainRange;
maxSeed    = seedRange(2);

clearvars -except isTest trainRange testRange seedRange maxSeed
clc; addpath('.\func\');
masterSeed = 1026;
rng(masterSeed,'combRecursive');
if isempty(gcp('nocreate')), parpool('threads'); end

%% 2. 全局常量 & 可调参数 ----------------------------------------------------
CDIFF_IDX      = 12;      % C.diff 索引
HEALTHY_THRESH = 1e-5;    % C.diff 健康上限
DISEASE_THRESH = 0.5;     % C.diff 疾病下限
DONOR_NZ_MAX   = 20;      % 供体稀疏约束
RECIP_NZ_MAX   = 43;      % 受体稀疏约束
zero_eps       = 1e-9;    % 极小截断

% —— 终态爆炸阈值 -----------------------------------------------------------
global MAX_ABUND_THRESH
MAX_ABUND_THRESH = 5;      % 若需更严，可减小

num_recipients       = 500;      % 目标成功受体数
donors_per_recipient = 40;      % 每受体需 40 供体
batchSize            = 500;     % 每轮并行候选数
maxBatches           = 2000;    % 供体批次上限

N              = 53;            % 物种数
time           = 0:0.1:30;
num_time_steps = numel(time);

output_dir = '.\train_globalR_500_4w_1026';
if ~exist(output_dir,'dir'), mkdir(output_dir); end

%% 3. 读取模型常量 -----------------------------------------------------------
A   = readmatrix("fake_A_53_dHOMO.csv");
A_c = parallel.pool.Constant(A);
T_c = parallel.pool.Constant(time);

% —— 全局统一生长率 (长度 53) ----------------------------------------------
r_global =[0.546393682769112, 0.760433327590399, 0.0353683323691972, 0.974659046968858, 0.731807335391887, 0.0710462328454989, 0.613604090023761, 0.834371235488301, 0.126546903200044, 0.351623621919953, 0.576280255262565, 0.790223729862463, 0.891626888086389, 0.772899504539558, 0.0986095500572747, 0.420563315102411, 0.0506158387011274, 0.483520458174648, 0.672017485748202, 0.0934335632165327, 0.817447862948387, 0.128099409461152, 0.207226049121578, 0.976336430477113, 0.0584022140987492, 0.838636627039298, 0.0948175681803862, 0.119278286710997, 0.0491398723843669, 0.981760843813893, 0.226782961438554, 0.468696080563501, 0.869907586826040, 0.631771781423754, 0.421126267492159, 0.890388402891514, 0.203172252659994, 0.991435713250166, 0.916212036673806, 0.913836269628040, 0.333467410006688, 0.378921990452184, 0.911069611511989, 0.0141651713399514, 0.546293708505807, 0.425619411235411, 0.313143856305284, 0.858090795614043, 0.848109146766575, 0.0729100072579711, 0.961585416834993, 0.114327988406995, 0.194123580993973];


%% 4. 初始化 HDF5 ------------------------------------------------------------
h5file = fullfile(output_dir,'FMT_structured_data.h5');
if exist(h5file,'file'), delete(h5file); end
h5create(h5file,'/time',[num_time_steps 1]); h5write(h5file,'/time',time');
h5create(h5file,'/metadata/species_index',[N 1],'Datatype','int32');
h5write(h5file,'/metadata/species_index',int32((1:N)'));
h5writeatt(h5file,'/metadata/species_index','Cdiff_index',CDIFF_IDX);

%% 5. Stage‑1+2：生成受体 & 供体 -------------------------------------------
successful          = 0;
recipient_ids       = strings(0);
disease_time_series = {};
recipient_r_list    = [];
donor_pool          = struct([]);
C_diff_matrix_tmp   = [];

donorTemplate = struct('id',"" ,'donor_final',zeros(N,1),'data',zeros(num_time_steps,N), ...
                       'status',int8(0),'seed',uint32(0),'tag',"" ,'ts_t',zeros(num_time_steps,N), ...
                       'ok',false,'c_final',nan);

numStreams  = num_recipients + 10;
global_used = false(maxSeed,1);

while successful < num_recipients
    idx = successful + 1;
    rid = sprintf('R%03d',idx);
    fprintf('\n[Process] 生成受体 %s  (已成功 %d / %d)\n',rid,successful,num_recipients);

    %% 5.1 受体生成 ---------------------------------------------------------
    seed_j = uint32(randi(seedRange));
    while true
        ws = RandStream.create('Threefry','Seed',seed_j,'NumStreams',numStreams,'StreamIndices',1);
        RandStream.setGlobalStream(ws);
        x0 = rand(ws,N,1); x0(CDIFF_IDX) = 0.01;

        [ts_R,endv_R,okR] = safe_glv(@G_disease_RK4_3,x0,A_c.Value,r_global,T_c.Value, ...
                                     CDIFF_IDX,seed_j,zero_eps);

        if okR && sum(endv_R<zero_eps,'all')<=RECIP_NZ_MAX && endv_R(CDIFF_IDX)>=DISEASE_THRESH
            break;   % 合格受体
        end
        seed_j = uint32(randi(seedRange));
    end

    % —— 写受体数据 --------------------------------------------------------
    dset = ['/recipients/' rid '/data']; delete_h5_path(h5file,dset);
    h5create(h5file,dset,size(ts_R')); h5write(h5file,dset,ts_R');

    dset_r = ['/recipients/' rid '/r']; delete_h5_path(h5file,dset_r);
    h5create(h5file,dset_r,[N 1]); h5write(h5file,dset_r,r_global.');

    recipient_ids(end+1)       = rid;
    disease_time_series{end+1} = ts_R';
    recipient_r_list(:,end+1)  = r_global.';

    %% 5.2 供体生成 ---------------------------------------------------------
    R_state    = endv_R;
    buf        = 0;    batch = 0;
    local_used = false(maxSeed,1);
    donor_buf  = repmat(donorTemplate,1,donors_per_recipient);
    cnt = struct('eff_noC',0,'eff_withC',0,'ineff_noC',0,'ineff_withC',0);
    tgt = struct('eff_noC',15,'eff_withC',5,'ineff_noC',15,'ineff_withC',5);

    while buf < donors_per_recipient && batch < maxBatches
        batch = batch + 1;
        parfor i = 1:batchSize
            candidates(i) = trialDonor(N,R_state,A_c.Value,r_global,T_c.Value, ...
                                       CDIFF_IDX,zero_eps,HEALTHY_THRESH,DONOR_NZ_MAX, ...
                                       DISEASE_THRESH,seedRange);
        end
        for i = 1:batchSize
            c = candidates(i);
            if ~c.ok || cnt.(c.tag)>=tgt.(c.tag), continue; end
            s = c.seed; if local_used(s)||global_used(s), continue; end

            buf = buf + 1; donor_buf(buf) = orderfields(c,donorTemplate);
            local_used(s) = true; cnt.(c.tag) = cnt.(c.tag)+1;

            tpath = sprintf('/transplants/%s/%s/data',c.id,rid); delete_h5_path(h5file,tpath);
            h5create(h5file,tpath,size(c.ts_t)); h5write(h5file,tpath,c.ts_t);
            fprintf('   + %s 终态C.diff=%.3g 标签=%s\n',c.id,c.c_final,c.tag);

            if buf>=donors_per_recipient, break; end
        end
        clear candidates
    end

    %% 5.3 失败 → 删除节点并重试 -------------------------------------------
    if buf < donors_per_recipient
        warning('%s 仅有 %d 供体，丢弃重试…',rid,buf);
        try
            h5rm_group(h5file,['/recipients/' rid]);
            info = h5info(h5file,'/transplants');
            for g = info.Groups
                h5rm_group(h5file,[g.Name '/' rid]);
            end
        end
        recipient_ids(end)       = [];
        disease_time_series(end) = [];
        recipient_r_list(:,end)  = [];
        continue;
    end

    %% 5.4 成功：记录 global_used & 累积 -----------------------------------
    for kk = 1:donors_per_recipient, global_used(donor_buf(kk).seed) = true; end
    successful = successful + 1;
    donor_pool = [donor_pool, donor_buf];
    C_diff_matrix_tmp(successful,:) = [donor_buf.status];
end

%% 6. Stage‑3：导出 HDF5 & CSV （与旧版一致，略）
fprintf('\n[Stage-3] 导出数据…\n');

donor_ids_all   = {donor_pool.id}';
donor_final_all = reshape([donor_pool.donor_final],N,[]);

[donor_ids_unique, ia, ic] = unique(donor_ids_all,'stable');
donor_ids   = matlab.lang.makeUniqueStrings(donor_ids_unique);
donor_final = donor_final_all(:, ia);

num_donors  = numel(donor_ids);
C_diff_labels = NaN(successful,num_donors);
for r = 1:successful
    cols = ic((r-1)*donors_per_recipient + (1:donors_per_recipient));
    C_diff_labels(r,cols) = C_diff_matrix_tmp(r,:);
end

% 写 HDF5 标签
h5create(h5file,'/transplants/donor_ids',[num_donors 1],'Datatype','string');
h5write (h5file,'/transplants/donor_ids',donor_ids);
h5create(h5file,'/transplants/recipient_ids',[successful 1],'Datatype','string');
h5write (h5file,'/transplants/recipient_ids',recipient_ids');
h5create(h5file,'/transplants/C_diff_labels',size(C_diff_labels));
h5write (h5file,'/transplants/C_diff_labels',C_diff_labels);

% 写 CSV 报表
csv_dir = fullfile(output_dir,'csv_reports');
if ~exist(csv_dir,'dir'), mkdir(csv_dir); end
species_hdr = arrayfun(@(x)sprintf('S%02d',x),1:N,'UniformOutput',false);

writetable(array2table(donor_final,'RowNames',species_hdr,'VariableNames',donor_ids),...
    fullfile(csv_dir,'donor_final_abundance.csv'),'WriteRowNames',true);

recip_final = cell2mat(cellfun(@(ts)ts(:,end),disease_time_series,'UniformOutput',false));
writetable(array2table(recip_final,'RowNames',species_hdr,'VariableNames',recipient_ids),...
    fullfile(csv_dir,'recipient_final_abundance.csv'),'WriteRowNames',true);

% —— 计算并写出 c_diff_real & 生长率 CSV —— 
c_final_matrix = reshape([donor_pool.c_final], donors_per_recipient, [])';  
c_diff_real = NaN(successful, num_donors);
for r = 1:successful
    cols = ic((r-1)*donors_per_recipient + (1:donors_per_recipient));
    c_diff_real(r,cols) = c_final_matrix(r,:);
end
writetable(array2table(c_diff_real,'RowNames',recipient_ids,'VariableNames',donor_ids),...
    fullfile(csv_dir,'transplant_cdiff_real.csv'),'WriteRowNames',true);

writetable(array2table(recipient_r_list,'RowNames',species_hdr,'VariableNames',recipient_ids),...
    fullfile(csv_dir,'recipient_growth_rates.csv'),'WriteRowNames',true);

writematrix(C_diff_labels, fullfile(csv_dir,'Cdiff_labels.csv'));
writetable(array2table(C_diff_labels,'RowNames',recipient_ids,'VariableNames',donor_ids),...
    fullfile(csv_dir,'Cdiff_labels_table.csv'),'WriteRowNames',true);

fprintf('全部完成！（模式：%s，成功受体数：%d）\n', ternStr(isTest,'TEST','TRAIN'), successful);

%% ================= 辅助函数 =================
function out = ternStr(cond,a,b)
    if cond, out = a; else, out = b; end
end
%% ================= 辅助函数 ================= ------------------------------
%% ---------- 生成供体：trialDonor ------------------------------------------
function cand = trialDonor(N,R_state,A,r_rec,time, ...
                           CDIFF_IDX,zero_eps,HEALTHY_THRESH,DONOR_NZ_MAX, ...
                           DISEASE_THRESH,seedRange)
% 生成单个供体，返回结构体 cand。若任何判定不合格，cand.ok = false。
% -------------------------------------------------------------------------

    % === 0) 结构体初始化 ===
    cand = struct('id',"",'donor_final',[],'data',[],'status',int8(0),'seed',uint32(0), ...
                  'tag',"",'ts_t',[],'ok',false,'c_final',nan);
    % === 1) 独立随机流 & 初值 ===
    seed   = randi(seedRange,'uint32');
    sDonor = RandStream('mt19937ar','Seed',seed);

    x0 = rand(sDonor,N,1);                   % U(0,1)
    k  = round(rand(sDonor)*0.30*N);         % 稀疏 0–30 %
    if k>0, x0(randperm(N,k)) = 0; end

    cdiff_init    = rand(sDonor) < 0.5;      % 是否带少量 C.diff
    x0(CDIFF_IDX) = cdiff_init*0.01;

    % === 2) 供体自身稳态 ===
    [~,d_end,ok1] = safe_glv(@glv_RK4_type,x0,A,r_rec,time,[],0,zero_eps);
    if ~ok1                              % (a) 出现 NaN 或终态爆炸
        return;
    end
    if sum(d_end < zero_eps,'all') > DONOR_NZ_MAX   % (b) 物种过少
        return;
    end
    if d_end(CDIFF_IDX) >= HEALTHY_THRESH           % (c) C.diff 仍过高
        return;
    end

    % === 3) 移植到受体后的动力学 ===
    [ts,x_post,ok2] = safe_glv(@glv_RK4_type,R_state+d_end, ...
                               A,r_rec,time,[],0,zero_eps);
    if ~ok2            % NaN / 爆炸
        return;
    end

    c_final = x_post(CDIFF_IDX);

    % === 4) 有效 / 无效 / 丢弃 ===
    if c_final < HEALTHY_THRESH
        status = 1;                         % 有效
    elseif c_final > DISEASE_THRESH
        status = 0;                         % 无效
    else
        return;                             % 中间区间 → 直接丢弃
    end

    % === 5) 标签 ===
    if     status==1 && ~cdiff_init, tag = 'eff_noC';
    elseif status==1 &&  cdiff_init, tag = 'eff_withC';
    elseif status==0 && ~cdiff_init, tag = 'ineff_noC';
    else                              tag = 'ineff_withC';
    end

    % === 6) 封装返回 ===
    cand.ok          = true;
    cand.id          = sprintf('D%06d',seed);
    cand.donor_final = d_end;
    cand.data        = ts;
    cand.status      = int8(status);
    cand.seed        = seed;
    cand.tag         = tag;
    cand.ts_t        = ts;
    cand.c_final     = c_final;
end

function [ts,x_end,ok] = safe_glv(f,x0,A,r,time,Cdiff,seed,zero_eps)
% safe_glv —— 包装 GLV 积分，统一质量检测
    if isequal(func2str(f),'glv_RK4_type')
        [ts,x_end] = f(x0,A,r,time,1,0.1,0.1);
    else
        [ts,x_end] = f(x0,A,r,time,1,0.1,0.1,Cdiff,0.01,seed);
    end

    global MAX_ABUND_THRESH
    has_nan  = any(isnan(ts(:)));
    has_blow = max(x_end,[],'all') > MAX_ABUND_THRESH;
    ok       = ~(has_nan | has_blow);
end

function h5rm_group(file,grpPath)
% 递归删除 HDF5 组（支持含子组）
    fid = H5F.open(file,'H5F_ACC_RDWR','H5P_DEFAULT');
    try
        gid = H5G.open(fid,grpPath);
        idx = 0;
        while true
            try
                name = H5L.get_name_by_idx(fid,grpPath,'H5_INDEX_NAME','H5_ITER_INC',idx,'H5P_DEFAULT');
            catch, break;
            end
            fullPath = strcat(grpPath,'/',name);
            info = H5O.get_info_by_name(fid,fullPath,'H5P_DEFAULT');
            if info.type == H5ML.get_constant_value('H5O_TYPE_GROUP')
                h5rm_group(file,fullPath);
            else
                H5L.delete(fid,fullPath,'H5P_DEFAULT');
            end
            idx = idx + 1;
        end
        H5L.delete(fid,grpPath,'H5P_DEFAULT');
        H5G.close(gid);
    end
    H5F.close(fid);
end

% function delete_h5_path(h5file,path)
% % 删除任意 HDF5 链接（dataset or group）
%     fid = H5F.open(h5file,'H5F_ACC_RDWR','H5P_DEFAULT');
%     try
%         H5L.delete(fid,path,'H5P_DEFAULT'); catch, end
%     H5F.close(fid);
% end
function delete_h5_path(h5file, path)
    % 删除 HDF5 文件里指定的链接（dataset 或 group）
    fid = H5F.open(h5file, 'H5F_ACC_RDWR', 'H5P_DEFAULT');
    try
        H5L.delete(fid, path, 'H5P_DEFAULT');
    catch
        % 如果不是链接（或已删除），忽略错误
    end
    H5F.close(fid);
end

