function [XX_disease, X_disease] = G_disease_RK4_2(initial, A, r, time, FunctionType, h1, h2, Cdiff, Cdiff_abundance, seed)
    % 生成符合条件的受体疾病状态
    oldRng = rng(seed); % 固定随机种子
    
    N = size(A, 1);
    max_attempts = 10000;
    success = false;
    
    for attempt = 1:max_attempts
        % 随机选择要删除的物种数量，并删除35-40个物种（保留C.diff）
        num_delete = randi([35,40]);
        deleted_species = setdiff(randsample(1:N, num_delete), Cdiff);
        
        temp_initial = initial;
        temp_initial(deleted_species) = 0;
        temp_initial(Cdiff) = Cdiff_abundance; % 强制设置C.diff
        
        % 运行动力学模型
        [XX, X] = glv_RK4_type(temp_initial, A, r, time, FunctionType, h1, h2);
        
        % 验证C.diff终态丰度>0.5
        if X(Cdiff) > 0.5
            XX_disease = XX;
            X_disease = X;
            success = true;
            break;
        end
    end
    
    if ~success
        warning('受体%d生成失败，使用NaN填充', seed);
        XX_disease = NaN(size(XX));
        X_disease = NaN(size(X));
    end
    rng(oldRng); % 恢复随机状态
end
