function w_optimal = minVar_TargetReturn(mu_target, mu, sigma)
n = length(mu); % Number of assets
A = [2*sigma, mu, ones(n,1); mu', 0, 0; ones(1,n), 0, 0];
b = [zeros(n,1); mu_target; 1];
solution = A \ b;
w_optimal = solution(1:n); % Optimal weights
end