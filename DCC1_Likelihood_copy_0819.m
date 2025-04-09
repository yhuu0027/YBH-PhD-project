function negLL = DCC1_Likelihood_copy_0819(params, et_sim)

    alpha_dcc = params(1);
    beta_dcc = params(2);

    % Initialize variables
    [T,N] = size(et_sim); 
    Q = zeros(N,N,T);

    % Find Si, the unconditional covariance matrix
   
    Si = et_sim'*et_sim ./T;
    Q(:,:,1) = Si;
    diagQ = diag(sqrt(diag(Q(:,:,1))));
    invSqrtDiagQ = diagQ^(-1);
    
    % initialize the value for Rt when t=1
    Rt(:,:,1) = invSqrtDiagQ * Q(:,:,1) * invSqrtDiagQ;

% Compute the log-likelihood
LL = -0.5 * (log(det(Rt(:,:,1))) + et_sim(1,:) * (inv(Rt(:,:,1))) *et_sim(1,:)');

for t = 2:T
    Q(:,:,t) = (1 - alpha_dcc - beta_dcc)*Si + alpha_dcc*(et_sim(t-1,:)'*et_sim(t-1,:)) + beta_dcc * Q(:,:,t-1);

    diagQ = diag(sqrt(diag(Q(:,:,t))));
    invSqrtDiagQ = inv(diagQ);
    Rt(:,:,t) = invSqrtDiagQ * Q(:,:,t) * invSqrtDiagQ;

    con_ll = -0.5 * (log(det(Rt(:,:,t))) + et_sim(t,:) * (inv(Rt(:,:,t))) *et_sim(t,:)');
 
    LL = LL + con_ll;
end
    
negLL  = - LL;
end


