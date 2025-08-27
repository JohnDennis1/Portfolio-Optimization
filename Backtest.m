function [portfolio_values, cumulative_returns, portfolio_volatilities] = Backtest(files, wt_opt, Rebalance_Freq)
% Backtest function to compute portfolio values, cumulative returns, and volatilities
% for multiple tickers.

% Initialize output variables
num_files = length(files);
portfolio_values = cell(num_files, 1);    % To store portfolio values for each ticker
cumulative_returns = zeros(num_files, 1); % To store cumulative returns for each ticker
portfolio_volatilities = zeros(num_files, 1); % To store volatilities for each ticker

% Loop through each file and perform calculations
for i = 1:num_files
    % Read data for the current ticker
    data = readmatrix(files(i));
    
    % Extract closing prices
    closing_prices = data(:, 5); % Assuming column 5 contains closing prices
    
    % Calculate returns (logarithmic)
    returns_backtest = diff(log(closing_prices)); % Logarithmic returns
    
    % Compute portfolio returns for the current ticker
    portfolio_returns = returns_backtest * wt_opt(i); % Weighted returns for the ticker
    
    % Compute cumulative portfolio value
    portfolio_value = cumprod(1 + portfolio_returns); % Starting value = 1
    
    % Store results
    portfolio_values{i} = portfolio_value; % Store the cumulative value series
    cumulative_returns(i) = portfolio_value(end) / portfolio_value(1) - 1; % Cumulative return
    portfolio_volatilities(i) = std(portfolio_returns) * sqrt(Rebalance_Freq); % Annualized volatility
end

end