
clear all;

%% simulate data
% yt = x1t*betat1 + x2t*betat2 + et
% et = sigmat*epsilont, where epsilont ~ N(0,1)

T= 1500;               % amount of total observations
T_in = 0.7*T;          % amount of in-sample data
T_out = T-T_in;            % amount of out-of-sample data
tau = (1:T)' / T;      % Time index tau = t/T
tau_in = tau(1:T_in);
p=2;                   % amount of predictors

%% define a Epanechnikov Kernel function 
EpanechnikovKernel = @(u) (3/4) * (1 - u.^2) .* (abs(u) <= 1);

%% generate x1t and x2t as they both have GJR-GARCH(1,1,1) structure

% x1t
mu1= 0;
omega1 = 0.03;
alpha1 = 0.0082;
gamma1 = 0.19;
beta1 = 0.8779;

% Set the seed for reproducibility
seed1 = 123;
rng(seed1);
epsilon_1 = randn(T, 1);
epsilon2_1 = epsilon_1.^2;
sigma2_x1 = zeros(T,1);
sigma2_x1(1) = omega1 / (1 - alpha1 - 0.5*gamma1 - beta1);

for t = 2:T
   if epsilon_1(t-1) >= 0
      sigma2_x1(t) = omega1 + alpha1 * epsilon2_1(t-1) + beta1 * sigma2_x1(t-1);
   else
      sigma2_x1(t) = omega1 + alpha1 * epsilon2_1(t-1) + gamma1 * epsilon2_1(t-1) + beta1 * sigma2_x1(t-1);
   end
end
   sigma_x1 = sqrt(sigma2_x1);

seed2 = 234;
rng(seed2);
u1t=randn(T,1);
for t = 1:T
    u1t(t) = epsilon_1(t) * sigma_x1(t);
end
x1t = mu1 + u1t; 


% x2t
mu2= 0;
omega2 = 0.02;
alpha2 = 0.0445;
gamma2 = 0.05;
beta2 = 0.926;

seed3 = 234567;
rng(seed3);
epsilon_2 = randn(T, 1);
epsilon2_2 = epsilon_2.^2;

sigma2_x2 = zeros(T,1);
sigma2_x2(1) = omega2 / (1 - alpha2 - 0.5*gamma2 - beta2);



sigma2_x2(2) = omega2 / (1 - alpha2 - 0.5*gamma2 - beta2);
for t = 3:T
    sigma2_x2(t) = omega2 + alpha2 * epsilon2_2(t-1)^2 + alpha2 * epsilon2_2(t-2)^2 + beta2 * sigma2_x2(t-1);
end
sigma_x2 = sqrt(sigma2_x2);

seed4 = 456;
rng(seed4);
u2t=randn(T,1);
for t = 1:T
    u2t(t) = epsilon_2(t) * sigma_x2(t);
end
x2t = mu2 + u2t; 

Xt = [x1t, x2t];

% Subplot for x1t
subplot(2,1,1);
plot(tau, x1t); % Plot x1t in blue
xlabel('\tau_t');
ylabel('x1t');
title('Time Series x1t');
grid on;

% Subplot for x2t
subplot(2,1,2);
plot(tau, x2t, 'r', 'LineWidth', 1); % Plot x2t in red
xlabel('\tau_t');
ylabel('x2t');
title('Time Series x2t');
grid on;

%% heterskedastic disturbance

seed5 = 2884;
rng(seed5);
epsilon_error = randn(T, 1);
epsilon2_error = epsilon_error.^2;


% Design - GARCH-like Deterministic Models. For example, GJR-GARCH(1,1,1)
% intuition: Volatility depends on past squared returns, but without stochastic errors
sigma2_error = zeros(T,1);

omega_error = 0.01;
alpha_error = 0.03;
gamma_error = 0.1;
beta_error = 0.9;

sigma2_error(1) = omega_error / (1 - alpha_error - 0.5*gamma_error - beta_error);
for t = 2:T
   if epsilon_error(t-1) >= 0
      sigma2_error(t) = omega_error + alpha_error * epsilon2_error(t-1) + beta_error * sigma2_error(t-1);
   else
      sigma2_error(t) = omega_error + alpha_error * epsilon2_error(t-1) + gamma_error * epsilon2_error(t-1) + beta_error * sigma2_error(t-1);
   end
end
   sigma_error = sqrt(sigma2_error);

seed6 = 678;
rng(seed6);
et=randn(T,1);
for t = 1:T
    et(t) = epsilon_error(t) * sigma_error(t);
end

% Plot et
figure;
plot(tau,et);
title('et');
xlabel('\tau');
ylabel('et');

% Plot volatility
figure;
plot(1:T, sigma_error, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Volatility');
xlabel('\tau');
ylabel('Volatility');
legend('Location', 'best');
grid on;



%% generate time-varying beta

betat =zeros(T,p);

% Design (piecewise function of time)
for t = 1:T
    % Compute beta1,t
    if tau(t) <= 0.32
        betat(t,1) = 1;
    elseif tau(t) <= 0.4
        betat(t,1) = 12.5 * tau(t) -3;
    elseif tau(t) <= 0.45
        betat(t,1) = -8 * tau(t) +5.2;
    else
        betat(t,1) = -2 * tau(t) + 2.5;
    end

    % Compute beta2,t
    if tau(t) <= 0.3
        betat(t,2) = -0.2;
    elseif tau(t) <= 0.4
        betat(t,2) = -2 * tau(t) + 0.4;
    elseif tau(t) <= 0.6
        betat(t,2) = 4 * tau(t) - 2;
    elseif tau(t) <= 0.8
        betat(t,2) = -2 * tau(t) + 1.6;
    else
        betat(t,2) = 17.5 * tau(t)^2 - 28 * tau(t) + 11.2;
    end
end

% Plot the time series
figure;
subplot(2,1,1);
plot(tau, betat(:,1), 'b', 'LineWidth', 1.5);
xlabel('\tau_t');
ylabel('\beta_{1,t}');
title('Time Series \beta_{1,t}');
grid on;
subplot(2,1,2);
plot(tau, betat(:,2), 'b', 'LineWidth', 1.5);
xlabel('\tau_t');
ylabel('\beta_{2,t}');
title('Time Series \beta_{2,t}');
grid on;




%% yt = beta1t*xit + beta2t*x2t+et
Yt = sum(Xt.*betat,2)+et;


% Plot et and yt
figure;
subplot(2,1,1)
plot(tau,et);
title('et');
xlabel('\tau');
ylabel('et');
hold on;
subplot(2,1,2)
plot(tau,Yt);
title('Yt');
xlabel('\tau');
ylabel('Yt');

% Plot yt
figure;
plot(tau,Yt);
title('Yt');
xlabel('\tau');
ylabel('Yt');

%% split the data to training vs testing set
Yt_estimation =Yt(1:T_in);       % training set
Xt_estimation =Xt(1:T_in,:);     % testing set



%% In-sample estimation

%% run benckmark model, DCC-GARCH, first
sigma2_dcc = zeros(T_in,1);   % set the dimension for the variance of the 10 assets
rt = Yt_estimation;
logLikelihood = @(params) GJR_GARCH_Likelihood(params, rt, T_in);
initialParams = [0.1, 0.1, 0.1, 0.1]; % Initial guess for [omega, alpha, gamma, beta]
lb = [0, 0, 0, 0];    % Lower bounds (e.g., alpha >= 0)
ub = [];              % No upper bounds
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');
estimatedParams = fmincon(logLikelihood, initialParams, [], [], [], [], lb, ub, [], options);
omega = estimatedParams(1);
alpha = estimatedParams(2);
gamma = estimatedParams(3);
beta = estimatedParams(4);

epsilon_dcc = rt - mean(rt); 
epsilon2_dcc = epsilon_dcc .^ 2;

sigma2_dcc(1) = omega / (1 - alpha - 0.5*gamma - beta);

for t = 2:T_in
 if epsilon_dcc(t-1) >= 0
 sigma2_dcc(t) = omega + alpha * epsilon2_dcc(t-1) + beta * sigma2_dcc(t-1);
 else
 sigma2_dcc(t) = omega + alpha * epsilon2_dcc(t-1) + gamma * epsilon2_dcc(t-1) + beta * sigma2_dcc(t-1);
 end
end

 sigma_dcc = sqrt(sigma2_dcc);

% GJR-GARCH for factors
factor = [x1t(1:T_in), x2t(1:T_in)];
[T_in,N] = size(factor);

sigma_factor = zeros(T_in,N);    % set the dimension for the volatility of the 10 assets
sigma2_factor = zeros(T_in,N);   % set the dimension for the variance of the 10 assets

epsilon_factor = zeros(T_in,N);  % innovation
epsilon2_factor = zeros(T_in,N); % innovation^2, for the garch structure

omegacollection_factor = zeros(N);
alphacollection_factor = zeros(N);
gammacollection_factor = zeros(N);
betacollection_factor = zeros(N);

for i=1:N
rt = factor(:,i);

% Define the log-likelihood function
logLikelihood = @(params) GJR_GARCH_Likelihood(params, rt, T_in);

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

for t = 2:T_in
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


% Find the Dt_dcc
rt = [Yt_estimation, x1t(1:T_in), x2t(1:T_in)];
[T_in,N] = size(rt);

% We construct the 3-dimension Dt matrix, which is the volatility matrix
Dt_dcc = zeros(N,N,T_in);
SIGMA = [sigma_dcc, sigma_factor(1:T_in,:)];

% Loop over each slice
for t = 1:T_in
    % Assign the diagonal of the t-th slice of Dt to the t-th row of SIGMA
    Dt_dcc(:,:,t) = diag(SIGMA(t, :));
end

et = rt ./ SIGMA; %volatility-adjusted return
Si = et'*et ./T_in; % Find Si, the unconditional covariance matrix

Q = zeros(N,N,T_in);
Q(:,:,1) = Si;
Rt = zeros(N,N,T_in);
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
for t = 2:T_in
    Q(:,:,t) = (1 - a_Q - b_Q)*Si + a_Q*(et(t-1,:)'*et(t-1,:)) + b_Q*Q(:,:,t-1);
    diagQ = diag(sqrt(diag(Q(:,:,t))));
    invSqrtDiagQ = inv(diagQ);
    Rt(:,:,t) = invSqrtDiagQ * Q(:,:,t) * invSqrtDiagQ;
end

betahat_dcc = zeros(2,T_in);
rho_sigma_vector = zeros(2,1);

% Loop through each time step
for t = 1:T_in
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

    betahat_dcc(:,t) = (Sigma_matrix)^(-1) * rho_sigma_vector;
end

% Store the betas
beta_market = betahat_dcc(1, :)';
beta_climate = betahat_dcc(2, :)';


% Display the estimated DCC parameters
disp('Estimated DCC Parameters:');
disp(['a_Q: ', num2str(a_Q)]);
disp(['b_Q: ', num2str(b_Q)]);


% Plot the results
figure;
subplot(2,1,1);
plot(tau_in, betat(1:T_in,1), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real \beta_1(\tau)');
hold on;
plot(tau_in, beta_market, 'r', 'LineWidth', 1.5,'DisplayName', 'DCC estimator \beta_1(\tau)'); 
hold on;
%fill([tau fliplr(tau)], [ci_lower_beta1_dcc' fliplr(ci_upper_beta1_dcc')], 'r', 'FaceAlpha', 0.2, 'DisplayName','95% confidence interval');
legend('Location', 'best');
title('Real vs DCC-estimated Beta1');
xlabel('\tau');
ylabel('\beta_1(\tau)');

subplot(2,1,2);
plot(tau_in, betat(1:T_in,2), 'b', 'LineWidth', 1.5, 'DisplayName', 'Real \beta_2(\tau)');
hold on;
plot(tau_in, beta_climate, 'r', 'LineWidth', 1.5, 'DisplayName', 'DCC estimator \beta_2(\tau)'); 
hold on;
legend('Location', 'best');
%fill([tau' fliplr(tau')], [ci_lower_beta2_new fliplr(ci_upper_beta2_new)], 'r', 'FaceAlpha', 0.2, 'DisplayName', '95% confidence interval');
title('Real vs DCC-estimated Beta2');
xlabel('\tau');
ylabel('\beta_2(\tau)');


%% Step 1: To find optimal bandwidth (leave many)

l =3;      % Define the size of the window to leave out (2l + 1 observations)
h_values = linspace(0.01, 0.99, 50);      % Define a range of bandwidth values
cv_errors = zeros(length(h_values), 1);   % Store CV errors for each bandwidth


% Leave (2l+1)-out cross-validation loop
for h_idx = 1:length(h_values)
    h = h_values(h_idx);     % Current bandwidth
    cv_error = 0;            % Initialize CV error for the current h
 
 for t = 1:2:T_in
 % Determine the indices to exclude (2l+1)-out around the t-th point
 exclude_indices = max(1, t-l):min(T_in, t+l); % Determine indices to exclude around point t
 
 % Create the leave-(2l+1)-out sample excluding the defined indices
 yt = Yt_estimation(setdiff(1:T_in, exclude_indices));        % Exclude the selected indices
 xt = Xt_estimation(setdiff(1:T_in, exclude_indices), :);     % Exclude the selected indices
 taut = tau(setdiff(1:T_in, exclude_indices));                % Exclude the selected indices
 
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
 cv_errors(h_idx) = cv_error / T_in;
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
hold on;
xline(h_old_optimal, '--k', 'optimal h')
xlabel('Bandwidth (h)');
ylabel('Cross-Validation Error');
title('Cross-Validation Error vs Bandwidth');
grid on;




%% Step 2: To obtain the residual series

% Fit the TVLM model via kernel (local constant), without assuming heteroskedastic error
beta_hat_old = zeros(T_in,2);
for t = 1:T_in
    u = (tau_in - tau_in(t))/h_old_optimal;  % Normalized distances
    weights = EpanechnikovKernel(u);     % Kernel weights
    W = diag(weights);                   % Diagonal weight matrix
    XWX = Xt_estimation' * W * Xt_estimation;
    XWy = Xt_estimation' * W * Yt_estimation;
    beta_hat_old(t,:) = XWX \ XWy;
end

% plot the estimated old_time-varying beta, in comparison with the true DGP
figure;
subplot(2, 1, 1);
plot(tau_in, beta_hat_old(:,1),'r','DisplayName', 'old estimated \beta_1(\tau)'); 
hold on;
plot(tau_in, betat(1:T_in,1), 'b', 'LineWidth', 1.5,'DisplayName', 'real \beta_1(\tau)');
title('real vs old estimation of \beta_1(\tau)');
legend('Location', 'best');
xlabel('\tau');
ylabel('\beta_1(\tau)');

subplot(2, 1, 2);
plot(tau_in, beta_hat_old(:,2),'r','DisplayName', 'old estimated \beta_2(\tau)');
hold on;
plot(tau_in, betat(1:T_in,2), 'b', 'LineWidth', 1.5,'DisplayName', 'real \beta_2(\tau)');
title('real vs old estimation of \beta_2(\tau)');
legend('Location', 'best');
xlabel('\tau');
ylabel('\beta_2(\tau)');


% obtain residual 
ethat = zeros(T_in,1);
for t = 1:T_in
    ethat(t) = Yt_estimation(t) - Xt_estimation(t,:) * beta_hat_old(t,:)';
end

% Theoretically, the mean of a variable that is log(chi-square distributed
% random variable) is -1.27. 
Dt = zeros(T_in,1);
for t = 1:T_in
    Dt(t)= log(ethat(t)^2)+1.27;
end


% We double check "mu = -1.27" with simulated data.
% Generate a random variable z from a standard normal distribution
seed7 = 28845080;
rng(seed7);
z = randn(1000000, 1);
% Plot the histogram of z to visualize its distribution
figure;
histogram(z, 'Normalization', 'pdf');
xlabel('z');
ylabel('Probability Density');
title('Histogram of z (Standard Normal Distribution)');
grid on;
% Generate s as the square of z. chi-sq
chi = z.^2;
% Plot the histogram of s to visualize its distribution
figure;
histogram(chi, 'Normalization', 'pdf');
xlabel('z^2');
ylabel('Probability Density');
title('Histogram of chi (chi = z^2)');
grid on;
mu = mean(log(chi));
disp('When T=1,000,000, the value of mu is:');
disp(mu);

%% Step 4: Recover the time-varying standard deviation

% select a new bandwidth
l =3;                                           % Define the size of the window to leave out (2l + 1 observations)
h0_candidates = linspace(0.01, 0.99, 100);      % Define a range of bandwidth values
CV_scores = zeros(length(h0_candidates), 1);    % Store CV errors for each bandwidth

% Leave (2l+1)-out cross-validation loop
for i = 1:length(h0_candidates)
 h_0 = h0_candidates(i);     % Current bandwidth
 CV_score = 0;               % Initialize CV error for the current h
 
 for t = 1:2:T_in
 % Determine the indices to exclude (2l+1)-out around the t-th point
 exclude_indices = max(1, t-l):min(T_in, t+l); % Determine indices to exclude around point t
 
 % Create the leave-(2l+1)-out sample excluding the defined indices
 Dt_train = Dt(setdiff(1:T_in, exclude_indices));      % Exclude the selected indices
 tau_train = tau(setdiff(1:T_in, exclude_indices));    % Exclude the selected indices
 
 % Local weights using the Epanechnikov kernel
 u = (tau(t) - tau_train) /h_0;     % Normalized distances
 weights = EpanechnikovKernel(u);   % Kernel weights for leave-(2l+1)-out data
 numerator = sum(weights .* Dt_train);
 denominator = sum(weights);     
 g_hat = numerator / denominator;       
 CV_score = CV_score + (Dt(t) - g_hat)^2;
 end
    
 % Average CV score for this h0
 CV_scores(i) = CV_score / T_in;
end

% Find the h (round two two digits) that minimizes the cross-validation error
[~, min_idx] = min(CV_scores);
h0_optimal = round(h0_candidates(min_idx),2); h0_optimal=0.03;

% Display the optimal h
fprintf('The optimal h_0 is: %.4f\n', h0_optimal);

% Plot the CV scores
figure;
plot(h0_candidates, CV_scores, 'b-', 'LineWidth', 1.5);
hold on;
xline(h0_optimal, '--k', 'optimal h')
xlabel('Bandwidth h0');
ylabel('Cross-Validation Score');
title('Cross-Validation Score vs Bandwidth for g(tau)');
grid on;

%% Use local constant estimator, Epanechnikov Kernel and the optimal bandwidth found in step 1 
gtauhat = zeros(T_in,1);

% local constant
for t = 1:T_in
    u = (tau_in - tau_in(t))/h0_optimal; 
    weight_D = EpanechnikovKernel(u);     % Compute weights using the kernel
    numerator = sum(weight_D .* Dt);      % Numerator: weighted sum of y
    denominator = sum(weight_D);          % Denominator: sum of weights
    gtauhat(t) = numerator / denominator;
end

% plot gtauhat (local linear estimator) together with Dt
figure;
plot(tau_in, Dt, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Observed (D_t)'); hold on;
plot(tau_in, gtauhat, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Prediction (g(\tau))'); % Add a horizontal line for g_tau_hat = 0.1649
xlabel('Time Index (t)');
ylabel('Values');
title('Observed vs. In-Sample Prediction');
legend('Location', 'best');
grid on;

% recover the standard deviation
sigmahat = exp(0.5 * gtauhat);

% Plot sigmahat with tau on the x-axis
figure;
plot(tau_in, sigmahat, 'r-', 'LineWidth', 1.5, 'DisplayName', 'estimated sigma(\tau)');
hold on;
plot(tau_in, sigma_error(1:T_in), 'b-', 'LineWidth', 1.5, 'DisplayName', 'sigma');
xlabel('\tau');
ylabel('true vs estimated Volatility');
legend('Location', 'best');
grid on;


%% compare estimated volatility
ethat_dcc = Yt_estimation - sum(Xt_estimation.*betahat_dcc',2);

% do GJR-GARCH to sigmahat_dcc.
sigma2 = zeros(T_in,1);   % set the dimension for the variance of the 10 assets
rt = ethat_dcc;
logLikelihood = @(params) GJR_GARCH_Likelihood(params, rt, T_in);
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
for t = 2:T_in
 if epsilon(t-1) >= 0
 sigma2(t) = omega + alpha * epsilon2(t-1) + beta * sigma2(t-1);
 else
 sigma2(t) = omega + alpha * epsilon2(t-1) + gamma * epsilon2(t-1) + beta * sigma2(t-1);
 end
end
 sigmahat_dcc = sqrt(sigma2);

figure;
plot(tau_in, sigma_error(1:T_in), 'b-', 'LineWidth', 1, 'DisplayName', 'true volatility');
hold on;
%plot(tau_in, sigmahat, 'r-', 'LineWidth', 1, 'DisplayName', 'estimated volatility from the proposed model');
%hold on;
plot(tau_in, sigmahat_dcc, 'g-', 'LineWidth', 1, 'DisplayName', 'estimated volatility from GARCH-DCC');
xlabel('\tau');
ylabel('Values');
legend('Location', 'best');
grid on;



%% Step 5: Re-estimate the time-varying beta with standardized xt and yt
beta_hat_new = zeros(T_in,2);
Xt_new = Xt_estimation ./ sigmahat;
Yt_new = Yt_estimation ./ sigmahat;

h_new_optimal = h_old_optimal; % use same bandwidth

% new estimator
for t = 1:T_in
    u = (tau_in - tau_in(t))/h_new_optimal;       % Normalized distances
    weights = EpanechnikovKernel(u);              % Kernel weights
    W = diag(weights);                            % Diagonal weight matrix
    XWX = Xt_new' * W * Xt_new;
    XWy = Xt_new' * W * Yt_new;
    beta_hat_new(t,:) = XWX \ XWy;
end

% plot the new estimate of time-varying betas vs real betas
subplot(2, 1, 1);
plot(tau_in, beta_hat_new(:,1), 'r', 'LineWidth', 1,'DisplayName', 'new estimated \beta_1(\tau)');
hold on;
plot(tau_in, beta_hat_old(:,1), 'r--', 'LineWidth', 1,'DisplayName', 'old estimated \beta_1(\tau)');
hold on;
plot(tau_in, betat(1:T_in,1), 'b', 'LineWidth', 1,'DisplayName', 'real \beta_1(\tau)');
title('Real vs Estimated \beta_{1}(\tau)');
legend('Location', 'best');
xlabel('\tau');
ylabel('\beta_1(\tau)');

subplot(2, 1, 2);
plot(tau_in, beta_hat_new(:,2), 'r', 'LineWidth', 1, 'DisplayName', 'new estimated \beta_2(\tau)');
hold on;
plot(tau_in, beta_hat_old(:,2), 'r--', 'LineWidth', 1,'DisplayName', 'old estimated \beta_1(\tau)');
hold on;
plot(tau_in, betat(1:T_in,2), 'b', 'LineWidth', 1,'DisplayName', 'real \beta_2(\tau)');
title('Real vs Estimated \beta_{2}(\tau)');
legend('Location', 'best');
xlabel('\tau');
ylabel('\beta_2(\tau)');


%% Evaluate in-sample estimation performance
% Compare MSE, in terms of y
error_old = Yt_estimation - sum(Xt_estimation.*beta_hat_old,2);
MSE_old = error_old'*error_old/T_in;

error_new = Yt_estimation - sum(Xt_estimation.*beta_hat_new,2);
MSE_new = error_new'*error_new/T_in;

error_dcc = Yt_estimation - sum(Xt_estimation.*betahat_dcc',2);
MSE_dcc = error_dcc'*error_dcc/T_in;
    

% Display the MSE from old and new estimation
fprintf('The new in-sample MSE is: %.4f\n', MSE_new);
fprintf('The old in-sample MSE is: %.4f\n', MSE_old);
fprintf('The dcc in-sample MSE is: %.4f\n', MSE_dcc);


mse_old_dcc=100*(MSE_dcc - MSE_old)/abs(MSE_dcc);
mse_new_dcc=100*(MSE_dcc - MSE_old)/abs(MSE_dcc);

fprintf('Compared with dcc, the old estimator brings the in-sample MSE down by (percentage): %.4f\n', mse_old_dcc);
fprintf('Compared with dcc, the new estimator brings the in-sample MSE down by (percentage): %.4f\n', mse_new_dcc);

%% one-step ahead OOS forecasting
j = 1;                            % one-step ahead out-of-sample forecasting
R = T_out -j+1;                   % number of rolling windows
fprintf('The foreward-forecasting step is: %.4f\n', j);
fprintf('The number of rolling windows to obtain OOS Rsq is: %.4f\n', R);
Yt_out =Yt(T_in+j:T);             % testing set

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


% old 
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
    weights = EpanechnikovKernel(u);              % Kernel weights
    W = diag(weights);                            % Diagonal weight matrix
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
% check with OOS MSE
% create a series for a neighborhood of the selected optimal bandwidth.

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

%for t = 1:T_in
for t = T_in
    u = (tau_in - tau_in(t))/h_new_optimal;
    weights = EpanechnikovKernel(u);
    W = diag(weights);  
    XWX = Xt_in' * W * Xt_in;
    XWy = Xt_in' * W * Yt_in;
    beta_hat_old_tvcmhe(t,:) = XWX \ XWy;
end

% Residual 
ethat = zeros(T_in,1);
%for t = 1:T_in
for t = T_in
    ethat(t) = Yt_estimation(t) - Xt_estimation(t,:) * beta_hat_old_tvcmhe(t,:)';
end

% Auxiliary kernel estimation
Dt = zeros(T_in,1);
%for t = 1:T_in
for t = T_in
    Dt(t)= log(ethat(t)^2)+1.27;
end

% Recover the time-varying standard deviation
gtauhat = zeros(T_in,1);
% for t = 1:T_in
for t = T_in
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
%for t = 1:T_in
for t = T_in
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
        %for t = 1:T_in
        for t = T_in
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
        %for t = 1:T_in
        for t = T_in
            u = (tau_in - tau_in(t))/h0_optimal;
            weight_D = EpanechnikovKernel(u);
            gtauhat(t) = sum(weight_D .* Dt) / sum(weight_D);
        end
        sigmahat = exp(0.5 * gtauhat);
        
        % Second-stage beta estimation with standardized data
        Xt_new = Xt_in ./ sigmahat;
        Yt_new = Yt_in ./ sigmahat;
        %for t = 1:T_in
        for t = T_in
            u = (tau_in - tau_in(t))/h_test;
            weights = EpanechnikovKernel(u);
            W = diag(weights);
            XWX = Xt_new' * W * Xt_new;
            XWy = Xt_new' * W * Yt_new;
            beta_hat_new_temp(t,:) = XWX \ XWy;
        end
        
        % Store forecast
        Yt_forecast_temp(r) = Xt(T_in+r,:) * beta_hat_new_temp(end,:)';
    end
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

fprintf('R-sq_OOS (TVCM vs HM): %.4f\n', R2_oos_tvcm);
fprintf('R-sq_OOS (TVCMHE vs HM): %.4f\n', R2_oos_tvcmhe);
fprintf('R-sq_OOS (DCC-GARCH vs HM): %.4f\n', R2_oos_dcc);

oos1_old_dcc=100*(R2_oos_tvcm - R2_oos_dcc)/abs(R2_oos_dcc);
oos1_new_dcc=100*(R2_oos_tvcmhe - R2_oos_dcc)/abs(R2_oos_dcc);

fprintf('Compared with dcc, the old estimator increased the out-of-sample R square by (percentage): %.4f\n', oos1_old_dcc);
fprintf('Compared with dcc, the new estimator increased the out-of-sample R square by (percentage): %.4f\n', oos1_new_dcc);
