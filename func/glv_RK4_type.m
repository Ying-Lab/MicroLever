function [dx, dxx] = glv_RK4_type(x, A, r, time, Type, h1, h2)
% 四阶Runge-Kutta方法求解广义Lotka-Volterra模型
% 输入参数与原函数保持一致，仅修改求解方法

N = size(A, 1);
nt = length(time);
dt = time(2) - time(1);

dxx = x(:); % 确保初始状态为列向量
dx = zeros(N, nt);
dx(:,1) = dxx;

% 定义各类型的微分方程函数句柄
switch Type
    case 1
        deriv_func = @(x) x .* (A' * x + r');
    case 2
        deriv_func = @(x) x .* (A' * (x ./ (1 + h1*x)) + r');
    % case 3
    %     h1h2_matrix = h1 + h2*eye(N);
    %     deriv_func = @(x) x .* (A * (x ./ (1 + x' * h1h2_matrix))' + r');
    % case 4
    %     deriv_func = @(x) x .* (A' * (x ./ ((1 + h1*x) .* (1 + h2*x))) + r');

    case 3
        deriv_func = @(x) x .* (...
            sum(A' .* (x' ./ (1 + h1*x' + h2*x)), 2) + r'...
        );
        
    case 4
        deriv_func = @(x) x .* (...
            sum(A' .* (x' ./ ((1 + h1*x') .* (1 + h2*x))), 2) + r'...
        );
end

% 统一的RK4时间步进
for i = 2:nt
    k1 = dt * deriv_func(dxx);
    k2 = dt * deriv_func(dxx + 0.5*k1);
    k3 = dt * deriv_func(dxx + 0.5*k2);
    k4 = dt * deriv_func(dxx + k3);
    
    dxx = dxx + (k1 + 2*k2 + 2*k3 + k4)/6;
    dx(:,i) = dxx;
    
    % 强制非负约束（可选）
    dxx(dxx < 0) = 0;
end

dx = dx'; % 保持输出维度为时间×物种
end