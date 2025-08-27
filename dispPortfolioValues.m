function dispPortfolioValues(portfolio_values, wt_opt)
figure;
hold on;
for i = 1:length(portfolio_values)
    plot(portfolio_values{i});
end
% Calculate overall portfolio value
num_assets = length(portfolio_values);
num_days = length(portfolio_values{1});
overall_portfolio_value = zeros(num_days, 1);

for i = 1:num_assets
    overall_portfolio_value = overall_portfolio_value + wt_opt(i) * portfolio_values{i};
end

% Overlay overall portfolio value
plot(overall_portfolio_value, 'k-', 'LineWidth', 1.75); % Black line for overall portfolio

% Customize the plot
title('Portfolio Value and Individual Ticker Performances');
xlabel('Time');
ylabel('Portfolio Value');
legend({'TSM', 'NVDA', 'HD', 'WFC', 'KO', 'Overall Portfolio'}, 'Location', 'Best');
grid on;
%print('MktValues_OptWts', '-dpng', '-r300')
end