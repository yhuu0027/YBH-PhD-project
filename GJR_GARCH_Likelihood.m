function [negLL,sigma2] = GJR_GARCH_Likelihood(params, rt, T)
    omega = params(1);
    alpha = params(2);
    gamma = params(3);
    beta = params(4);
    
    % Initialize variables
    sigma2 = zeros(T, 1);
    epsilon = rt - mean(rt);    % epsilon is observed according to the constant-mean GARCH structure.
    epsilon2 = epsilon .^ 2;
    
    % Initialize the value of sigma based on epsilon(1)
    sigma2(1) = omega / (1 - alpha - 0.5*gamma - beta);
 
    % Compute the log-likelihood
    LL = -0.5 * log(sigma2(1)) - epsilon2(1) / (2*sigma2(1));
    
    for t = 2:T
        if epsilon(t-1) >= 0
            sigma2(t) = omega + alpha * epsilon2(t-1) + beta * sigma2(t-1);
        else
            sigma2(t) = omega + alpha * epsilon2(t-1) + gamma * epsilon2(t-1) + beta * sigma2(t-1);
        end
        
        con_ll = -0.5 * log(sigma2(t)) - epsilon2(t) / (2*sigma2(t));
        LL = LL + con_ll;
    end
    
    % Return the negative log-likelihood
    negLL = -LL; % Negative because fminunc minimizes the function
end