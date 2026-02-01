function [XX_disease, X_disease] = G_disease_RK4_3(initial, A, r, time, ...
        FunctionType, h1, h2, Cdiff, Cdiff_abundance, seed)
% 生成符合条件的受体疾病状态
%
% - 若传入 seed (非空)，用该 seed 固定随机序列；
% - 若 seed 为空或未提供，则沿用调用方当前 RNG，不做任何重置。

    % ---------- 1. 处理随机种子 ----------
    if nargin < 10,  seed = [];  end              % 允许省略 seed
    oldRng = rng;                                 % 记录原 RNG 状态
    if ~isempty(seed)
        rng(seed);                                % 仅在 seed 有效时才设
    end

    % ---------- 2. 主体逻辑 ----------
    N             = size(A, 1);
    max_attempts  = 10000;
    success       = false;

    for attempt = 1:max_attempts
        % 随机删除 35–40 个物种（保留 C.diff）
        num_delete      = randi([35, 40]);
        deleted_species = setdiff(randsample(N, num_delete), Cdiff);

        temp_initial           = initial;
        temp_initial(deleted_species) = 0;
        temp_initial(Cdiff)    = Cdiff_abundance;   % 设置 C.diff

        % 运行动力学模型
        [XX, X] = glv_RK4_type(temp_initial, A, r, time, FunctionType, h1, h2);

        % 终态 C.diff 丰度判定
        if X(Cdiff) > 0.5
            XX_disease = XX;
            X_disease  = X;
            success    = true;
            break;
        end
    end

    % ---------- 3. 若失败则返回 NaN ----------
    if ~success
        warning('受体(种子=%s) 生成失败，返回 NaN', mat2str(seed));
        XX_disease = NaN(size(XX));
        X_disease  = NaN(size(X));
    end

    % ---------- 4. 恢复 RNG 状态 ----------
    rng(oldRng);
end
