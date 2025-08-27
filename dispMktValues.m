function dispMktValues(files)
    % Load data from files
    data1 = readmatrix(files(1));
    data2 = readmatrix(files(2));
    data3 = readmatrix(files(3));
    data4 = readmatrix(files(4));
    data5 = readmatrix(files(5));

    % Extract close price data
    close_asset1 = data1(:, 5);
    close_asset2 = data2(:, 5);
    close_asset3 = data3(:, 5);
    close_asset4 = data4(:, 5);
    close_asset5 = data5(:, 5);

    % Combine into a single matrix for easier operations
    portfolio_values = [close_asset1, close_asset2, close_asset3, close_asset4, close_asset5];

    % Calculate portfolio value over time
    overall_portfolio_value = sum(portfolio_values, 2);

    % Plot individual asset close prices
    figure;
    hold on;
    plot(close_asset1, 'LineWidth', 1.25);
    plot(close_asset2, 'LineWidth', 1.25);
    plot(close_asset3, 'LineWidth', 1.25);
    plot(close_asset4, 'LineWidth', 1.25);
    plot(close_asset5, 'LineWidth', 1.25);

    % Overlay overall portfolio value
    plot(overall_portfolio_value, 'k-', 'LineWidth', 1.75); % Black line for overall portfolio value

    % Customize the plot
    title('Real Asset Values Over Time');
    xlabel('Time');
    ylabel('Portfolio and Asset Values');
    legend('TSM', 'NVDA', 'HD', 'WFC', 'KO', 'Overall Portfolio', 'Location', 'Best');
    grid on;
    hold off;
    %print('MktValues', '-dpng', '-r300')
end