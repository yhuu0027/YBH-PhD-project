clear all;

% Dimensions
T= 5432;               % amount of total observations
T_out = 1630;            % amount of out-of-sample data
T_in = T - T_out;      % amount of in-sample data
tau = (1:T)' / T;      % Time index tau = t/T
tau_in = tau(1:T_in);
p=2;                   % number of predictors
N = 10;                % number of US banks

% define a Epanechnikov Kernel function 
EpanechnikovKernel = @(u) (3/4) * (1 - u.^2) .* (abs(u) <= 1);

%% Import data
% real daily stock return
filePath = '/Users/yuehenghu/Desktop/research/Climate Risk';
fileName = 'dailyreturn(1).xlsx';
fullFileName = fullfile(filePath, fileName);
data = readtable(fullFileName);
r_t=zeros(T,N); % stock return in a T*N matrix
for i= 1 : N
r_t(:,i)=100 * data{105:5536,i+1};
end
Yt = r_t(:,1);

filename = '/Users/yuehenghu/Desktop/research//Climate Risk/F-F_Research_Data_Factors_daily.xlsx';
data1 = readtable(filename);
rf = data1{105:5536, 5};
z_t = r_t - rf; % excess stock return
mkt = data1{105:5536, 2}; % market excess return

% the climate factor:
filename2 = '/Users/yuehenghu/Desktop/research/Climate Risk/KOL_XLE0828.xlsx';
data2 = readtable(filename2);
kol = data2{105:5536, 2};
xle = data2{105:5536, 3};
sta = 100*(0.7*kol + 0.3*xle); %The stranded asset factor is composed of a 70% KOL + 30% xle
cf = sta; %The stranded asset is used as an proxy of the climate factor. 

% rename variables to be consistent to simulation study
x1t = mkt;
x2t = cf;
Xt =[x1t,x2t];

%% plot x1t, x2t and yt
figure;

subplot(3,1,1);
plot(tau, Yt, 'b', 'LineWidth', 1.5); 
hold on;
%legend('Location', 'best');
title('yt');
xlabel('\tau');
ylabel('value');
hold on;

subplot(3,1,2);
plot(tau, x1t, 'b', 'LineWidth', 1.5); 
hold on;
%legend('Location', 'best');
title('x1t');
xlabel('\tau');
ylabel('value');
hold on;

subplot(3,1,3);
plot(tau, x2t, 'b', 'LineWidth', 1.5); 
hold on;
%legend('Location', 'best');
title('x2t');
xlabel('\tau');
ylabel('value');


%% split the data to estimation vs testing set
Yt_estimation =Yt(1:T_in);       % estimation set
Xt_estimation =Xt(1:T_in,:);     % testing set


%% In-sample estimation. 

% To find optimal bandwidth (leave many) with full sample

l =3;      % Define the size of the window to leave out (2l + 1 observations)
h_values = linspace(0.01, 0.99, 50);      % Define a range of bandwidth values
cv_errors = zeros(length(h_values), 1);   % Store CV errors for each bandwidth

% Leave (2l+1)-out cross-validation loop
for h_idx = 1:length(h_values)
    h = h_values(h_idx);     % Current bandwidth
    cv_error = 0;            % Initialize CV error for the current h

 for t = 1:2:T
 % Determine the indices to exclude (2l+1)-out around the t-th point
 exclude_indices = max(1, t-l):min(T, t+l); % Determine indices to exclude around point t

 % Create the leave-(2l+1)-out sample excluding the defined indices
 yt = Yt(setdiff(1:T, exclude_indices));        % Exclude the selected indices
 xt = Xt(setdiff(1:T, exclude_indices), :);     % Exclude the selected indices
 taut = tau(setdiff(1:T, exclude_indices));                % Exclude the selected indices

 % Local weights using the Epanechnikov kernel
 u = (taut - tau(t)) / h; % Normalized distances
 weights = EpanechnikovKernel(u); % Kernel weights for leave-(2l+1)-out data

 % Weighted least squares to estimate beta at tau(t) excluding (2l+1) neighbors
 W = diag(weights); % Diagonal weight matrix
 XWX = xt' * W * xt;
 XWy = xt' * W * yt;

 % Solve for beta estimates using the weighted least squares solution
 beta_hat_minus_t = XWX \ XWy;
 Yt_pred = Xt(t, :) * beta_hat_minus_t;
 cv_error = cv_error + (Yt(t) - Yt_pred)^2;
 end

 % Average the CV error for the current h
 cv_errors(h_idx) = cv_error / T;
end

% Find the h that minimizes the cross-validation error
[~, min_idx] = min(cv_errors);
h_old_optimal = h_values(min_idx);
h_old_optimal = round(h_old_optimal, 2);  % Round to 2 decimal place

% Display the optimal h
fprintf('The optimal h is: %.4f\n', h_old_optimal);

% Plot cv_errors against h_values
figure;
plot(h_values, cv_errors, 'b-o', 'LineWidth', 2);
xlabel('Bandwidth (h)');
ylabel('Cross-Validation Error');
title('Cross-Validation Error vs Bandwidth');
grid on;

% use this bandwidth for all kernel estimation
h0_optimal = h_old_optimal;
h_new_optimal = h_old_optimal;

% Step 2: To obtain the residual series
% Fit the TVLM model via kernel (local constant), without assuming heteroskedastic error
beta_hat_old = zeros(T,2);
for t = 1:T
    u = (tau - tau(t))/h_old_optimal;  % Normalized distances
    weights = EpanechnikovKernel(u);  % Kernel weights
    W = diag(weights);  % Diagonal weight matrix
    XWX = Xt' * W * Xt;
    XWy = Xt' * W * Yt;
    beta_hat_old(t,:) = XWX \ XWy;
end

% plot the estimated time-varying beta, in comparison with the true DGP
figure;
subplot(2, 1, 1);
plot(tau, beta_hat_old(:,1),'r','DisplayName', 'old estimation of \beta_1(\tau)'); 
title('Old estimation of \beta_1(\tau)');
legend('Location', 'best');
xlabel('\tau');
ylabel('\beta_1(\tau)');
subplot(2, 1, 2);
plot(tau, beta_hat_old(:,2),'r','DisplayName', 'old estimation of \beta_2(\tau)');
title('Old estimation of \beta_2(\tau)');
legend('Location', 'best');
xlabel('\tau');
ylabel('\beta_2(\tau)');


% obtain residual 
ethat = zeros(T,1);
for t = 1:T
    ethat(t) = Yt(t) - Xt(t,:) * beta_hat_old(t,:)';
end

% plot residual
figure;
plot(1:T, ethat, 'b-', 'LineWidth', 1.5, 'DisplayName', 'residual e_t hat');
xlabel('\tau');
ylabel('Values');
legend('Location', 'best');
grid on;

% Auxiliary regression with the residual as dependent variable
Dt = zeros(T,1);
for t = 1:T
    Dt(t)= log(ethat(t)^2)+1.27;
end

%Recover the time-varying standard deviation
gtauhat = zeros(T,1);
% local constant
for j = 1:T
    u = (tau - tau(j))/h0_optimal; 
    % Compute weights using the kernel
    weight_D = EpanechnikovKernel(u);  
    % Numerator: weighted sum of y
    numerator = sum(weight_D .* Dt);
    % Denominator: sum of weights
    denominator = sum(weight_D);
    gtauhat(j) = numerator / denominator;
end

% plot gtauhat (local linear estimator) together with Dt
figure;
plot(1:T, Dt, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Observed (D_t)'); hold on;
plot(1:T, gtauhat, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Prediction (g(\tau))');
% Add a horizontal line for g_tau_hat = 0.1649
xlabel('Time Index (t)');
ylabel('Values');
title('Observed vs. In-Sample Prediction');
legend('Location', 'best');
grid on;
% recover the standard deviation
sigmahat = exp(0.5 * gtauhat);

% Plot sigmahat with tau on the x-axis
figure;
plot(tau, sigmahat, 'b-', 'LineWidth', 1.5, 'DisplayName', 'estimated sigma(\tau)');
xlabel('\tau');
ylabel('Values');
legend('Location', 'best');
grid on;

% Re-estimate the time-varying beta with standardized xt and yt
beta_hat_new = zeros(T,2);
Xt_new = Xt ./ sigmahat;
Yt_new = Yt ./ sigmahat;

for t = 1:T
    u = (tau - tau(t))/h_new_optimal;             % Normalized distances
    weights = EpanechnikovKernel(u);  % Kernel weights
    W = diag(weights);                % Diagonal weight matrix
    XWX = Xt_new' * W * Xt_new;
    XWy = Xt_new' * W * Yt_new;
    beta_hat_new(t,:) = XWX \ XWy;
end

% plot the new estimate of time-varying betas vs real betas
subplot(2, 1, 1);
plot(tau, beta_hat_new(:,1), 'r', 'LineWidth', 1.5,'DisplayName', 'New estimated \beta_1(\tau)');
title('New Estimated \beta_{1}(\tau)');
legend('Location', 'best');
xlabel('\tau');
ylabel('\beta_1(\tau)');
subplot(2, 1, 2);
plot(tau, beta_hat_new(:,2), 'r', 'LineWidth', 1.5, 'DisplayName', 'New estimated \beta_2(\tau)');
title('New Estimated \beta_{2}(\tau)');
legend('Location', 'best');
xlabel('\tau');
ylabel('\beta_2(\tau)');

%% Full-sample estimation, DCC-GARCH
% GJR-GARCH for dependent variable
sigma2 = zeros(T,1);   % set the dimension for the variance of the 10 assets
rt = Yt;
logLikelihood = @(params) GJR_GARCH_Likelihood(params, rt, T);
initialParams = [0.1, 0.1, 0.1, 0.1]; % Initial guess for [omega, alpha, gamma, beta]
lb = [0, 0, 0, 0]; % Lower bounds (e.g., alpha >= 0)
ub = []; % No upper bounds
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
estimatedParams = fmincon(logLikelihood, initialParams, [], [], [], [], lb, ub, [], options);
omega = estimatedParams(1);
alpha = estimatedParams(2);
gamma = estimatedParams(3);
beta = estimatedParams(4);
epsilon = rt - mean(rt); 
epsilon2 = epsilon .^ 2;
sigma2(1) = omega / (1 - alpha - 0.5*gamma - beta);
for t = 2:T
 if epsilon(t-1) >= 0
 sigma2(t) = omega + alpha * epsilon2(t-1) + beta * sigma2(t-1);
 else
 sigma2(t) = omega + alpha * epsilon2(t-1) + gamma * epsilon2(t-1) + beta * sigma2(t-1);
 end
end
 sigma = sqrt(sigma2);

% GJR-GARCH for factors
factor = [x1t, x2t];
[T,N] = size(factor);
sigma_factor = zeros(T,N);    % set the dimension for the volatility of the 10 assets
sigma2_factor = zeros(T,N);   % set the dimension for the variance of the 10 assets
epsilon_factor = zeros(T,N);  % innovation
epsilon2_factor = zeros(T,N); % innovation^2, for the garch structure
omegacollection_factor = zeros(N);
alphacollection_factor = zeros(N);
gammacollection_factor = zeros(N);
betacollection_factor = zeros(N);
for i=1:N
rt = factor(:,i);
% Define the log-likelihood function
logLikelihood = @(params) GJR_GARCH_Likelihood(params, rt, T);
% Set initial values for omega, alpha, and beta
initialParams = [0.1, 0.1, 0.1, 0.1]; % Initial guess for [omega, alpha, gamma, beta]
lb = [0, 0, 0, 0]; % Lower bounds (e.g., alpha >= 0)
ub = []; % No upper bounds
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
estimatedParams = fmincon(logLikelihood, initialParams, [], [], [], [], lb, ub, [], options);
omega_factor = estimatedParams(1);
alpha_factor = estimatedParams(2);
gamma_factor = estimatedParams(3);
beta_factor = estimatedParams(4);
epsilon_factor(:,i) = rt - mean(rt);    % epsilon is observed according to the constant-mean GARCH structure.
epsilon2_factor(:,i) = epsilon_factor(:,i) .^ 2;
sigma2_factor(1,i)= omega / (1 - alpha - 0.5*gamma - beta);
for t = 2:T
    if epsilon_factor(t-1,i) >= 0
       sigma2_factor(t,i) = omega + alpha * epsilon2_factor(t-1,i) + beta * sigma2_factor(t-1,i);
    else
       sigma2_factor(t,i) = omega + alpha * epsilon2_factor(t-1,i) + gamma * epsilon2_factor(t-1,i) + beta * sigma2_factor(t-1,i);
    end
end
sigma_factor(:,i) = sqrt(sigma2_factor(:,i));

% store the estimated parameters
omegacollection_factor(i)= omega_factor;
alphacollection_factor(i)=alpha_factor;
gammacollection_factor(i)=gamma_factor;
betacollection_factor(i)=beta_factor;
end

% Find the Dt
rt = [Yt, x1t, x2t];
[T,N] = size(rt);
% We construct the 3-dimension Dt matrix, which is the volatility matrix
Dt = zeros(N,N,T);
SIGMA = [sigma, sigma_factor(1:T,:)];
% Loop over each slice
for t = 1:T
    % Assign the diagonal of the t-th slice of Dt to the t-th row of SIGMA
    Dt(:,:,t) = diag(SIGMA(t, :));
end
et = rt ./ SIGMA; %volatility-adjusted return
Si = et'*et ./T; % Find Si, the unconditional covariance matrix
Q = zeros(N,N,T);
Q(:,:,1) = Si;
Rt = zeros(N,N,T);
diagQ = diag(sqrt(diag(Q(:,:,1))));
invSqrtDiagQ = inv(diagQ);
Rt(:,:,1) = invSqrtDiagQ * Q(:,:,1) * invSqrtDiagQ;

% estimate the DCC coefficients (w.r.t Q)
options = optimset('fminunc'); 
options.Display = 'off';
options.LargeScale = 'off';
options.MaxIter = 1000;
dcc_logLikelihood = @(params) DCC1_Likelihood_copy_0819(params, et);
initial = [0.1, 0.1];

lb = [0, 0]; % Lower bounds (e.g., alpha >= 0)
ub = [1, 1]; % upper bound

options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
estimatedParams = fmincon(dcc_logLikelihood, initial, [], [], [], [], lb, ub, [], options);
a_Q = estimatedParams(1);
b_Q = estimatedParams(2);

% Predicted Q, then calculate Rt
for t = 2:T
    Q(:,:,t) = (1 - a_Q - b_Q)*Si + a_Q*(et(t-1,:)'*et(t-1,:)) + b_Q*Q(:,:,t-1);
    diagQ = diag(sqrt(diag(Q(:,:,t))));
    invSqrtDiagQ = inv(diagQ);
    Rt(:,:,t) = invSqrtDiagQ * Q(:,:,t) * invSqrtDiagQ;
end
betas = zeros(2,T);
rho_sigma_vector = zeros(2,1);
% Loop through each time step
for t = 1:T
    % Extract conditional variances
    sigma_it = SIGMA(t,1);
    sigma_mt = SIGMA(t,2);
    sigma_ct = SIGMA(t,3);
    % Extract conditional correlations from Rt
    rho_imt = Rt(1, 2, t);
    rho_ict = Rt(1, 3, t);
    rho_mct = Rt(2, 3, t);
    % Construct the matrices for the equation
    Sigma_matrix = [sigma_mt^2, rho_mct * sigma_mt * sigma_ct; 
                    rho_mct * sigma_mt * sigma_ct, sigma_ct^2];
    r1 = rho_imt * sigma_it * sigma_mt;
    r2 = rho_ict * sigma_it * sigma_ct;
    rho_sigma_vector(1) = r1;
    rho_sigma_vector(2) = r2;
    betas(:,t) = (Sigma_matrix)^(-1) * rho_sigma_vector;
end
% Store the betas
beta_market = betas(1, :)';
beta_climate = betas(2, :)';
% Display the estimated DCC parameters
disp('Estimated DCC Parameters:');
disp(['a_Q: ', num2str(a_Q)]);
disp(['b_Q: ', num2str(b_Q)]);

beta_dcc = betas';

% obtain residual and plot error term's volatility
ethat_dcc = Yt - sum(Xt.*betas',2);

% do GJR-GARCH to sigmahat_dcc.
sigma2 = zeros(T,1);   % set the dimension for the variance of the 10 assets
rt = ethat_dcc;
logLikelihood = @(params) GJR_GARCH_Likelihood(params, rt, T);
initialParams = [0.1, 0.1, 0.1, 0.1]; % Initial guess for [omega, alpha, gamma, beta]
lb = [0, 0, 0, 0]; % Lower bounds (e.g., alpha >= 0)
ub = []; % No upper bounds
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
estimatedParams = fmincon(logLikelihood, initialParams, [], [], [], [], lb, ub, [], options);
omega = estimatedParams(1);
alpha = estimatedParams(2);
gamma = estimatedParams(3);
beta = estimatedParams(4);
epsilon = rt - mean(rt); 
epsilon2 = epsilon .^ 2;
sigma2(1) = omega / (1 - alpha - 0.5*gamma - beta);

for t = 2:T
 if epsilon(t-1) >= 0
 sigma2(t) = omega + alpha * epsilon2(t-1) + beta * sigma2(t-1);
 else
 sigma2(t) = omega + alpha * epsilon2(t-1) + gamma * epsilon2(t-1) + beta * sigma2(t-1);
 end
end
 sigmahat_dcc = sqrt(sigma2);

% compare with the estimated volatility from the proposed model
figure;
plot(tau, sigmahat, 'b-', 'LineWidth', 1.5, 'DisplayName', 'estimated volatility from the proposed model');
hold on;
plot(tau, sigmahat_dcc, 'g-', 'LineWidth', 1.5, 'DisplayName', 'estimated volatility from GARCH-DCC');
xlabel('\tau');
ylabel('Values');
legend('Location', 'best');
grid on;




%% out-of-sample forecasting. Use out-of-sample R-sq (OOS Rsq) to evaluate model's performance. 
% one-step ahead forecasting
j = 1;                 % forecast step
R = T_out - j+1;       % how many rolling windows to obtain an average OOS performance
j = 1;                            % one-step ahead out-of-sample forecasting
fprintf('The foreward-forecasting step is: %.4f\n', j);
fprintf('The number of rolling windows to obtain OOS Rsq is: %.4f\n', R);
Yt_out =Yt(T_in+j:T_in+j+R-1);    % testing set

% benckmark model's prediction error first
% Historical Mean model
Yt_forecast_hm = zeros(R,1);
for r = 1:R
    % re-define the in-sample period for each window
    Yt_in = Yt(r:T_in+r-1);
    Yt_forecast_hm(r) = mean(Yt_in);  % in-sample period is from 1 to T_in
end
pred_error_hm = (Yt_out-Yt_forecast_hm)'*(Yt_out-Yt_forecast_hm);
% Display the oos prediction error
fprintf('The oos prediction error from HM is: %.4f\n', pred_error_hm);

%% old 
% Time Varying Coefficient Model (TVCM)

Yt_forecast_tvcm = zeros(R,1);
for r = 1:R
    % re-define the in-sample period for each window
    Yt_in = Yt(r:T_in+r-1);
    Xt_in = Xt(r:T_in+r-1,:);
    tau_in = (r:T_in+r-1)' / T_in;
   % Loop over each time point to estimate beta1(tau) and beta2(tau)
   % for t = 1:T_in
    u = (tau_in - tau_in(T_in))/h_old_optimal;    % Normalized distances
    weights = EpanechnikovKernel(u);       % Kernel weights
    W = diag(weights);                     % Diagonal weight matrix
    XWX = Xt_in' * W * Xt_in;
    XWy = Xt_in' * W * Yt_in;
    beta_hat = XWX \ XWy;
   % end
Yt_forecast_tvcm(r) = Xt(T_in+r,:) * beta_hat;
end

% Find the aggregate prediction error from the R windows (per in the OOS R-sq equation)
pred_error_tvcm = (Yt_out-Yt_forecast_tvcm)'*(Yt_out-Yt_forecast_tvcm);

% Out-of-sample R-square
R2_oos_tvcm = 1- pred_error_tvcm/pred_error_hm;

% Display the oos prediction error
fprintf('The oos prediction error from TVCM is: %.4f\n', pred_error_tvcm);

% Display the oos R-sq
fprintf('The oos R-sq from TVCM, in comparison with the HM model, is: %.4f\n', R2_oos_tvcm);

%% Bandwidth robustness check
% Create the neighborhood range
step = 0.01;                             % Step size for neighborhood
range0 = h_old_optimal + (-2:2)' * step; % vector centered at optimal_h
n_bandwidths0 = length(range0);
cv_mse0 = zeros(n_bandwidths0, 1);

% Initialize storage for prediction errors across bandwidths
pred_error_tvcm_all = zeros(n_bandwidths0, 1);
R2_oos_tvcm_all = zeros(n_bandwidths0, 1);

for i = 1:n_bandwidths0
    h = range0(i);
   % One-step-ahead forecasting with current bandwidth
    Yt_forecast_tvcm = zeros(R, 1);
    for r = 1:R
        % Define in-sample window
        Yt_in = Yt(r:T_in+r-1);
        Xt_in = Xt(r:T_in+r-1, :);
        tau_in = (r:T_in+r-1)' / T_in;

        % Estimate beta_hat for the last point in the window
        t_last = T_in;  % Focus on the last point (one-step-ahead forecast)
        u = (tau_in - tau_in(t_last)) / h;
        weights = EpanechnikovKernel(u);
        W = diag(weights);
        XWX = Xt_in' * W * Xt_in;
        XWy = Xt_in' * W * Yt_in;
        beta_hat = XWX \ XWy;

        % Forecast
        Yt_forecast_tvcm(r) = Xt(T_in + r, :) * beta_hat;
    end

    % Compute prediction error and R-squared
    pred_error_tvcm_all(i) = (Yt_out - Yt_forecast_tvcm)' * (Yt_out - Yt_forecast_tvcm);
    R2_oos_tvcm_all(i) = 1 - pred_error_tvcm_all(i) / pred_error_hm;

end

% Create a table to display bandwidth vs. R-squared
results_table_old = table(range0, R2_oos_tvcm_all, ...
    'VariableNames', {'Bandwidth', 'OOS_R_Squared'});

% Display the table (shows first 10 rows by default)
disp('Bandwidth vs. Out-of-Sample R^2 (old):');
disp(results_table_old);

% Optional: Export to CSV file
writetable(results_table_old, 'bandwidth_vs_r_squared_old.csv');


%% new 
% Time Varying Coefficient Model with Heterskedastic Error (TVCMHE)
Yt_forecast_tvcmhe = zeros(R,1);
beta_hat_old_tvcmhe = zeros(T_in,2);

for r = 1:R
    % re-define the in-sample period for each window
    Yt_in = Yt(r:T_in+r-1);
    Xt_in = Xt(r:T_in+r-1,:);
    tau_in = (r:T_in+r-1)' / T_in;

for t = 1:T_in
    u = (tau_in - tau_in(t))/h_new_optimal;
    weights = EpanechnikovKernel(u);
    W = diag(weights);  
    XWX = Xt_in' * W * Xt_in;
    XWy = Xt_in' * W * Yt_in;
    beta_hat_old_tvcmhe(t,:) = XWX \ XWy;
end

% Residual 
ethat = zeros(T_in,1);
for t = 1:T_in
    ethat(t) = Yt_estimation(t) - Xt_estimation(t,:) * beta_hat_old_tvcmhe(t,:)';
end

% Auxiliary kernel estimation
Dt = zeros(T_in,1);
for t = 1:T_in
    Dt(t)= log(ethat(t)^2)+1.27;
end

% Recover the time-varying standard deviation
gtauhat = zeros(T_in,1);
for t = 1:T_in
    u = (tau_in - tau_in(t))/h0_optimal; 
    weight_D = EpanechnikovKernel(u);  
    numerator = sum(weight_D .* Dt);
    denominator = sum(weight_D);
    gtauhat(t) = numerator / denominator;
end
sigmahat = exp(0.5 * gtauhat);

% Re-estimate the time-varying beta with standardized xt and yt
beta_hat_new_tvcmhe = zeros(T_in,2);
Xt_new = Xt_in ./ sigmahat;
Yt_new = Yt_in ./ sigmahat;
for t = 1:T_in
    u = (tau_in - tau_in(t))/h_new_optimal;        
    weights = EpanechnikovKernel(u);            
    W = diag(weights);                    
    XWX = Xt_new' * W * Xt_new;
    XWy = Xt_new' * W * Yt_new;
    beta_hat_new_tvcmhe(t,:) = XWX \ XWy;
end

Yt_forecast_tvcmhe(r) = Xt(T_in+r,:) * beta_hat_new_tvcmhe(end,:)';
end

% Find the aggregate prediction error from the R windows (per in the OOS R-sq equation)
pred_error_tvcmhe = (Yt_out-Yt_forecast_tvcmhe)'*(Yt_out-Yt_forecast_tvcmhe);

% Out-of-sample R-square
R2_oos_tvcmhe = 1- pred_error_tvcmhe/pred_error_hm;

% Display the oos prediction error
fprintf('The oos prediction error from TVCMHE is: %.4f\n', pred_error_tvcmhe);

% Display the oos R-sq
fprintf('The oos R-sq from TVCMHE, in comparison with the HM model, is: %.4f\n', R2_oos_tvcmhe);


% bandwidth robustness check
% Define neighborhood around optimal bandwidth
step = 0.01;                             % Step size for neighborhood
range1 = h_new_optimal + (-2:2)' * step; % bandwidth values centered at h_new_optimal
n_bandwidths1 = length(range1);

% Initialize storage for results
R2_oos_tvcmhe_all = zeros(n_bandwidths1, 1);
bandwidth_results = range1;

% Loop over bandwidth candidates
for i = 1:n_bandwidths1
    h_test = range1(i);

    % Temporary storage for this bandwidth
    Yt_forecast_temp = zeros(R,1);
    beta_hat_new_temp = zeros(T_in,2);

    for r = 1:R
        % Re-define in-sample period for each window
        Yt_in = Yt(r:T_in+r-1);
        Xt_in = Xt(r:T_in+r-1,:);
        tau_in = (r:T_in+r-1)' / T_in;

        % --- Standard TVCMHE estimation with current bandwidth ---

        % First-stage beta estimation
        beta_hat_old_temp = zeros(T_in,2);
        for t = 1:T_in
            u = (tau_in - tau_in(t))/h_test;
            weights = EpanechnikovKernel(u);
            W = diag(weights);
            XWX = Xt_in' * W * Xt_in;
            XWy = Xt_in' * W * Yt_in;
            beta_hat_old_temp(t,:) = XWX \ XWy;
        end

        % Residual calculation
        ethat = Yt_in - sum(Xt_in .* beta_hat_old_temp, 2);

        % Volatility estimation (using original h0_optimal)
        Dt = log(ethat.^2) + 1.27;
        gtauhat = zeros(T_in,1);
        for t = 1:T_in
            u = (tau_in - tau_in(t))/h0_optimal;
            weight_D = EpanechnikovKernel(u);
            gtauhat(t) = sum(weight_D .* Dt) / sum(weight_D);
        end
        sigmahat = exp(0.5 * gtauhat);

        % Second-stage beta estimation with standardized data
        Xt_new = Xt_in ./ sigmahat;
        Yt_new = Yt_in ./ sigmahat;
        for t = 1:T_in
            u = (tau_in - tau_in(t))/h_test;
            weights = EpanechnikovKernel(u);
            W = diag(weights);
            XWX = Xt_new' * W * Xt_new;
            XWy = Xt_new' * W * Yt_new;
            beta_hat_new_temp(t,:) = XWX \ XWy;
        end

        % Store forecast
        Yt_forecast_temp(r) = Xt(T_in+r-1,:) * beta_hat_new_temp(end,:)';
    end

    % Calculate OOS R-squared for this bandwidth
    pred_error_temp = (Yt_out-Yt_forecast_temp)'*(Yt_out-Yt_forecast_temp);
    R2_oos_tvcmhe_all(i) = 1 - pred_error_temp/pred_error_hm;
end

% Display results
results_table_new = table(bandwidth_results, R2_oos_tvcmhe_all, ...
    'VariableNames', {'Bandwidth', 'OOS_R2'});
disp(results_table_new);


% Optional: Export to CSV file
writetable(results_table_new, 'bandwidth_vs_r_squared_new.csv');



% Plot results
figure;
plot(bandwidth_results, R2_oos_tvcmhe_all, '-o', 'LineWidth', 1.5);
hold on;
plot(bandwidth_results, R2_oos_tvcm_all, 'bo-', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
hold on;
xline(h_old_optimal, '--k', 'optimal h from CV');
xlabel('Bandwidth (h)');
ylabel('Out-of-sample R^2');
title('OOS R^2 vs Bandwidth');
legend('new estimator', 'old estimator', 'Location', 'best');
grid on;



%% DCC
Yt_forecast_dcc = zeros(R,1);

for r = 1:R
% re-define the in-sample period for each window
Xt_in = Xt(r:T_in+r-1,:);
Yt_in = Yt(r:T_in+r-1,:);
% step 1: Update Conditional volatility (Dt_new)
% GJR-GARCH for excess return
rt = Yt_in;
logLikelihood = @(params) GJR_GARCH_Likelihood(params, rt, T_in);
initialParams = [0.1, 0.1, 0.1, 0.1]; 
lb = [0, 0, 0, 0];      % Lower bounds (e.g., alpha >= 0)
ub = [];                % No upper bounds
options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
estimatedParams = fmincon(logLikelihood, initialParams, [], [], [], [], lb, ub, [], options);
omega = estimatedParams(1);
alpha = estimatedParams(2);
gamma = estimatedParams(3);
beta = estimatedParams(4);
epsilon = rt - mean(rt);   
epsilon2 = epsilon .^ 2;
sigma2 = zeros(T_in,1);
sigma2(1) = omega / (1 - alpha - 0.5*gamma - beta);
for t = 2:T_in+1
   if epsilon(t-1) >= 0
      sigma2(t) = omega + alpha * epsilon2(t-1) + beta * sigma2(t-1);
   else
      sigma2(t) = omega + alpha * epsilon2(t-1) + gamma * epsilon2(t-1) + beta * sigma2(t-1);
   end
end
   sigma = sqrt(sigma2);
   sigma_new = sqrt(sigma(T_in+1)); % the volatility at time t+j

% GJR-GARCH for factors
factor = Xt_in;
[T,p] = size(factor);
sigma_factor = zeros(T_in+1,p); 
sigma2_factor = zeros(T_in+1,p);

for i=1:p
    rt = factor(:,i);
    logLikelihood = @(params) GJR_GARCH_Likelihood(params, rt, T);
    initialParams = [0.1, 0.1, 0.1, 0.1]; 
    options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
    estimatedParams = fminunc(logLikelihood, initialParams, options);
    omega_factor = estimatedParams(1);
    alpha_factor = estimatedParams(2);
    gamma_factor = estimatedParams(3);
    beta_factor = estimatedParams(4);
    epsilon_factor(:,i) = rt - mean(rt);    % epsilon is observed according to the constant-mean GARCH structure.
    epsilon2_factor(:,i) = epsilon_factor(:,i) .^ 2;
    sigma2_factor(1,i)= omega_factor / (1 - alpha_factor - 0.5*gamma_factor - beta_factor);

    for t = 2:T_in+1
        if epsilon_factor(t-1,i) >= 0
           sigma2_factor(t,i) = omega_factor + alpha_factor * epsilon2_factor(t-1,i) + beta_factor * sigma2_factor(t-1,i);
        else
           sigma2_factor(t,i) = omega_factor + alpha_factor * epsilon2_factor(t-1,i) + gamma_factor * epsilon2_factor(t-1,i) + beta_factor * sigma2_factor(t-1,i);
        end
    end

    sigma_factor(:,i) = sqrt(sigma2_factor(:,i));
    sigma2_factor_new = sigma2_factor(T_in+1,:);
end

sigma_factor_new = sqrt(sigma2_factor_new);
SIGMA = [sigma(1:T_in,:), sigma_factor(1:T_in,:)];
SIGMA_new = [sigma_new, sigma_factor_new];
Dt_new = diag(SIGMA_new);

% step 2: Forecasting the Correlation Matrix 
rt = [Yt_in,Xt_in];
[T,N]=size(rt);
et = rt ./ SIGMA;    % volatility-adjusted return
et_sim = et;
Si = et'*et ./T;     % Find Si, the unconditional covariance matrix
Q = zeros(N,N,T);
Q(:,:,1) = Si;
Rt = zeros(N,N,T);
diagQ = diag(sqrt(diag(Q(:,:,1))));
invSqrtDiagQ = inv(diagQ);
Rt(:,:,1) = invSqrtDiagQ * Q(:,:,1) * invSqrtDiagQ;

% estimate the DCC coefficients (w.r.t Q)
options = optimset('fminunc'); 
options.Display = 'off';
options.LargeScale = 'off';
options.MaxIter = 1000;
dcc_logLikelihood = @(params) DCC1_Likelihood_copy_0819(params, et_sim);
initial = [0.1, 0.1];
lb = [0, 0]; % Lower bounds (e.g., alpha >= 0)
ub = [1, 1]; % upper bound
options = optimoptions('fminunc', 'Display', 'iter', 'Algorithm', 'quasi-newton');
estimatedParams = fmincon(dcc_logLikelihood, initial, [], [], [], [], lb, ub, [], options);
a_Q = estimatedParams(1);
b_Q = estimatedParams(2);
Q_new = zeros(N,N,T_in+1); % psuedo-covarianc matrix; one-step ahead forecasting
Q_new(:,:,1) = Si;
for t = 2:T_in+1
    Q_new(:,:,t) = (1 - a_Q - b_Q)*Si + a_Q*(et(t-1,:)'*et(t-1,:)) + b_Q*Q_new(:,:,t-1);
    diagQ = diag(sqrt(diag(Q_new(:,:,t))));
    invSqrtDiagQ = inv(diagQ);
    Rt_new(:,:,t) = invSqrtDiagQ * Q_new(:,:,t) * invSqrtDiagQ;
end

% step 4: apply the closed-form expression of beta
rho_sigma_vector = zeros(p,1);
    % Extract conditional variances
    sigma_it = SIGMA_new(1);
    sigma_mt = SIGMA_new(2);
    sigma_ct = SIGMA_new(3);

    % Extract conditional correlations from Rt
    rho_imt = Rt_new(1, 2, T_in+1);
    rho_ict = Rt_new(1, 3, T_in+1);
    rho_mct = Rt_new(2, 3, T_in+1);

% Construct the matrices for the equation
    Sigma_matrix = [sigma_mt^2, rho_mct * sigma_mt * sigma_ct; 
                    rho_mct * sigma_mt * sigma_ct, sigma_ct^2];
    r1 = rho_imt * sigma_it * sigma_mt;
    r2 = rho_ict * sigma_it * sigma_ct;
    rho_sigma_vector(1) = r1;
    rho_sigma_vector(2) = r2;
    beta_oos_dcc = (Sigma_matrix)^(-1) * rho_sigma_vector;
    Yt_forecast_dcc(r) = Xt(T_in+r, :) * beta_oos_dcc; 
end


pred_error_dcc = (Yt_out-Yt_forecast_dcc)'*(Yt_out-Yt_forecast_dcc);

% Out-of-sample R-square
R2_oos_dcc = 1- pred_error_dcc/pred_error_hm;

% Display the oos prediction error
fprintf('The oos prediction error from DCC-GARCH is: %.4f\n', pred_error_dcc);

% Display the oos R-sq
fprintf('R-sq_OOS (DCC-GARCH vs HM): %.4f\n', R2_oos_dcc);

%%
fprintf('R-sq_OOS (TVCM vs HM): %.4f\n', R2_oos_tvcm);
fprintf('R-sq_OOS (TVCMHE vs HM): %.4f\n', R2_oos_tvcmhe);
fprintf('R-sq_OOS (DCC-GARCH vs HM): %.4f\n', R2_oos_dcc);

oos1_old_dcc=100*(R2_oos_tvcm - R2_oos_dcc)/abs(R2_oos_dcc);
oos1_new_dcc=100*(R2_oos_tvcmhe - R2_oos_dcc)/abs(R2_oos_dcc);

fprintf('Compared with dcc, the old estimator increased the out-of-sample R square by (percentage): %.4f\n', oos1_old_dcc);
fprintf('Compared with dcc, the new estimator increased the out-of-sample R square by (percentage): %.4f\n', oos1_new_dcc);







