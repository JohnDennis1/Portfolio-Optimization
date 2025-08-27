function m_optimal = GlobalOptimal(sigma, mu)
n = length(mu);
A = [2*sigma, ones(n,1); ones(1,n),0 ];
b = [0, 0, 0, 0, 0, 1]';
solution = A\b;
m_optimal = solution(1:n);
end