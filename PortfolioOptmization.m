clc;
clear
close all;

%Optimizing For Jan 3 2018 - Jan 2019 (Dec 29 2018)
%% Creating File names
file1 = "Mktdata_TSM.xlsx";
file2 = "Mktdata_NVDA.xlsx";
file3 = "Mktdata_HD.xlsx";
file4 = "Mktdata_WFC.xlsx";
file5 = "Mktdata_KO.xlsx";
% These files are for back-testing
fileTSM_2018 = "TSM_2018.xlsx";
fileNVDA_2018 = "NVDA_2018.xlsx";
fileHD_2018 = "HD_2018.xlsx";
fileWFC_2018 = "WFC_2018.xlsx";
fileKO_2018 = "KO_2018.xlsx";
%creating vectors for use in functions
files = [file1 file2 file3 file4 file5];
files2 = [fileTSM_2018 fileNVDA_2018 fileHD_2018 fileWFC_2018 fileKO_2018];

%% Setting up the portfolio with basic elements such as tickers, intial opening values, the RoR, weight, and the average close
stock1 = struct('name', 'TSM', 'value_i', 0, 'mu', 0, 'weight_i', 0.1205297629, 'mean_close', 0, "wt_opt", 0); %Taiwan Semiconductors
stock2 = struct('name', 'NVDA', 'value_i', 0, 'mu', 0, 'weight_i', 0.0162000811, 'mean_close', 0, "wt_opt", 0); %Nvidia
stock3 = struct('name', 'HD', 'value_i', 0, 'mu', 0, 'weight_i', 0.553324059, 'mean_close', 0, "wt_opt", 0); %Home Depot
stock4 = struct('name', 'WFC', 'value_i', 0, 'mu', 0, 'weight_i', 0.1768967716, 'mean_close', 0, "wt_opt", 0); % Wells Fargo and Co.
stock5 = struct('name', 'KO', 'value_i', 0, 'mu', 0, 'weight_i', 0.1330493248, 'mean_close', 0, "wt_opt", 0); %Coca-cola

%These functions simply read in historical market data and retrieve certain
%values such as intial stock value at the beginning of the year, the mean
%closing value for each asset, and it's RoR
[stock1.value_i, stock1.mean_close, stock1.mu] = getValues(fileTSM_2018);
[stock2.value_i, stock2.mean_close, stock2.mu] = getValues(fileNVDA_2018);
[stock3.value_i, stock3.mean_close, stock3.mu] = getValues(fileHD_2018);
[stock4.value_i, stock4.mean_close, stock4.mu] = getValues(fileWFC_2018);
[stock5.value_i, stock5.mean_close, stock5.mu] = getValues(fileKO_2018);

%% Calculating the Covariance Matrix, weight vector, portfolio expected return, and portfolio variance
mu = [stock1.mu stock2.mu stock3.mu stock4.mu stock5.mu]'; %Generating the expected returns vector containing each asset's expected return

[sigma, returns] = getCov(files); %Calculating the covariance matrix

weights = [stock1.weight_i stock2.weight_i stock3.weight_i stock4.weight_i stock5.weight_i]';%Creating a weight vector to contain all the weights of the assets

muPortfolio = weights' * mu;%Expected returns of the Portfolio

muPortfolio_annual = (1 + muPortfolio / 100)^252 - 1;%Annualizing it for 252 days, cumulative Returns

variancePortfolio = weights' * sigma * weights;%Variance of the portfolio


%% Global Minimum Variance
w_GMV = GlobalOptimal(sigma, mu);% Calculating the weights for the Global Minimum Variance
mu_GMV = dot(w_GMV,mu);%Expected return from the GMV
var_GMV = w_GMV' * sigma * w_GMV;%The value of the GMV
%% Solving Constrained optimization problems for minimizing variance with target return
mu_target_1 = 0.10;% Target for 10% return
mu_target_2 = .07;% Target for 7% return
X = minVar_TargetReturn(mu_target_1, mu, sigma); % The optimal weights for 10% return
w_target1 = minVar_TargetReturn(mu_target_1, mu, sigma);
mu_target1 = mu' * w_target1; % return of X/w_target1, they're the same
var_target1 = w_target1' * sigma * w_target1; % variance of X
Y = minVar_TargetReturn(mu_target_2, mu, sigma); % Optimal weights for target 2
w_target2 = minVar_TargetReturn(mu_target_2, mu, sigma);
mu_target2 = mu' * w_target2; % return of Y
var_target2 = w_target2' * sigma * w_target2; % variance of Y
stock1.wt_opt = X(1);
stock2.wt_opt = X(2);
stock3.wt_opt = X(3);
stock4.wt_opt = X(4);
stock5.wt_opt = X(5);
%% Portfolio
Portfolio = [stock1 stock2 stock3 stock4 stock5];% Setting up the portfolio

%% Efficient Frontier

alpha = linspace(-1,2,100);%Creating spacing for later use
mu_frontier = zeros(size(alpha)); 
sigma_frontier = zeros(size(alpha)); 

% z matrix
for i = 1:length(alpha)
    w = (alpha(i) * w_GMV) + ((1 - alpha(i)) * X); 
    mu_frontier(i) = mu' * w; 
    sigma_frontier(i) = sqrt(w' * sigma * w); 
end 

% plotting the efficeint frontier
figure; 
plot(sigma_frontier, mu_frontier, 'k-', 'LineWidth', 1.5);
hold on; 
plot(sqrt(var_target1), mu_target1, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r'); %Plotting the 10% portfolio
plot(sqrt(var_target2), mu_target2, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b'); %Plotting the 7% portfolio
%Formatting
xlabel('Portfolio Risk (Standard Deviation)')
ylabel('Portfolio Return')
title('Efficient Frontier')
legend('Efficient Frontier', '10% - Somewhat Unrealistic', ...
    '7% - Realistic', 'Location', 'Best');
ylim([-0.3, 0.3]);
xlim([0.006, .0137]);
grid on;
%print('Efficient_Frontier', '-dpng', '-r300')
hold off;


%% Back-Test for the year 2023
Rebalance_Freq = 252; %daily
[portfolio_value_2023, cumulative_return_2023, portfolio_variance_2023] = Backtest(files, X, Rebalance_Freq);% Back testing

%% Visualizing Results

dispPortfolioValues(portfolio_value_2023, X);

%% Displaying Real Market Values for 2023
dispMktValues(files);

%% PCA of variance for the problem

[eVecs, eVecs_raw, D, Percent_Impact, Feature, Var] = getPCA(sigma);


%% Scatter Plot of Returns vs Variance
dispMuVar(mu,Var);
%% Functions

function [open_i, mean_close, RoR] = getValues(filename)

data = readmatrix(filename);%Read in the Files

open_i = data(end, 2);%The opening value at the beginning of the year

fin_close = data(1,5);%The closing value at the end of the year

mean_close = mean(data(:,5));%The average closing values throughout the year

RoR = (fin_close-open_i)/open_i;%THe rate of return, expected return, or mu

end

function [sigma, returns] = getCov(files)
%Creating the variables containing the files
f1 = files(1);
f2 = files(2);
f3 = files(3);
f4 = files(4);
f5 = files(5);
%reading the files
data1 = readmatrix(f1);
data2 = readmatrix(f2);
data3 = readmatrix(f3);
data4 = readmatrix(f4);
data5 = readmatrix(f5);
%Gathering the closing values
close_asset1 = data1(:,5);
close_asset2 = data2(:,5);
close_asset3 = data3(:,5);
close_asset4 = data4(:,5);
close_asset5 = data5(:,5);
%Creating a vector of all the closing values
tot_close = [close_asset1 close_asset2 close_asset3 close_asset4 close_asset5];
%Calculating Percentage Returns
returns = diff(tot_close) ./ tot_close(1:end-1, :);
%Creating the covariance matrix with the returns
sigma = cov(returns);

end

function w_optimal = minVar_TargetReturn(mu_target, mu, sigma)
n = length(mu); % Number of assets
A = [2*sigma, mu, ones(n,1); mu', 0, 0; ones(1,n), 0, 0];%Creating the matrix to solve the system of equations
b = [zeros(n,1); mu_target; 1];%Defining the solution
solution = A \ b;%Calculating the solution
w_optimal = solution(1:n); % Optimal weights exlcuding the lagrange multipliers
end

function [portfolio_values, cumulative_returns, portfolio_variance] = Backtest(files, wt_opt, Rebalance_Freq)
% Backtesting function to compute portfolio values, cumulative returns, and volatilities
% for multiple tickers.

% Initialize output variables
num_files = length(files);
portfolio_values = cell(num_files, 1);    %For the overall portfolio values
cumulative_returns = zeros(num_files, 1); %For the cumulative returns
portfolio_variance = zeros(num_files, 1); %Storing the volatilites/variances of each asset

% Loop through each file and perform calculations
for i = 1:num_files
    %Read data
    data = readmatrix(files(i));
    
    %Grabs the closing 
    closing_prices = data(:, 5);

    returns_backtest = diff(log(closing_prices)); % Logarithmic returns
    
    % Computing portfolio returns for the asset considering its weight
    portfolio_returns = returns_backtest * wt_opt(i);
    
    % Computing the cumulative portfolio value in total
    portfolio_value = cumprod(1 + portfolio_returns);
    
    %Storing the values for later use
    portfolio_values{i} = portfolio_value; % Store the cumulative value series
    cumulative_returns(i) = portfolio_value(end) / portfolio_value(1) - 1; % Cumulative return
    portfolio_variance(i) = std(portfolio_returns) * sqrt(Rebalance_Freq); % Annualized variance
end

end

function dispPortfolioValues(portfolio_values, wt_opt)% Helps to visualize results of the portfolio_values
figure;
hold on;
for i = 1:length(portfolio_values)%Goes through the cells to plot it
    plot(portfolio_values{i});
end
% This calculates overall portfolio value
num_assets = length(portfolio_values); %used for the for loop%
num_days = length(portfolio_values{1});%Used for length
overall_portfolio_value = zeros(num_days, 1);

for i = 1:num_assets
    overall_portfolio_value = overall_portfolio_value + wt_opt(i) * portfolio_values{i};%Calculating the value of the portfolio
end

plot(overall_portfolio_value, 'k-', 'LineWidth', 1.75); % Black line for overall portfolio

% Formatting
title('Portfolio Value and Individual Ticker Performances');
xlabel('Time');
ylabel('Portfolio Value');
legend({'TSM', 'NVDA', 'HD', 'WFC', 'KO', 'Overall Portfolio'}, 'Location', 'Best');
grid on;
%print('MktValues_OptWts', '-dpng', '-r300')
end


function dispMktValues(files)
    % read data from files
    data1 = readmatrix(files(1));
    data2 = readmatrix(files(2));
    data3 = readmatrix(files(3));
    data4 = readmatrix(files(4));
    data5 = readmatrix(files(5));

    % Getting the close price data
    close_asset1 = flipud(data1(:, 5));
    close_asset2 = flipud(data2(:, 5));
    close_asset3 = flipud(data3(:, 5));
    close_asset4 = flipud(data4(:, 5));
    close_asset5 = flipud(data5(:, 5));

    % Making it into a vector
    portfolio_values = [close_asset1, close_asset2, close_asset3, close_asset4, close_asset5];

    % Calculating the portfolio value over all the values
    overall_portfolio_value = sum(portfolio_values, 2);

    %Plotting the values of each asset and then putting the black line for
    %the overall portfolio value
    figure;
    hold on;
    plot(close_asset1, 'LineWidth', 1.25);
    plot(close_asset2, 'LineWidth', 1.25);
    plot(close_asset3, 'LineWidth', 1.25);
    plot(close_asset4, 'LineWidth', 1.25);
    plot(close_asset5, 'LineWidth', 1.25);

    plot(overall_portfolio_value, 'k-', 'LineWidth', 1.75); % Black line for overall portfolio value

    %More Formatting
    title('Real Asset Values Over Time');
    xlabel('Time');
    ylabel('Portfolio and Asset Values');
    legend('TSM', 'NVDA', 'HD', 'WFC', 'KO', 'Overall Portfolio', 'Location', 'Best');
    grid on;
    hold off;
    print('MktValues', '-dpng', '-r300')
end

function [eVecs, eVecs_raw, D, Percent_Impact, Feature, Var] = getPCA(sigma)
    % eigenvalues and eigenvectors
    [eVecs_raw, e_raw] = eig(sigma);
    Var = diag(sigma); % Variances of each asset

    %Getting and thensort eigenvalues in descending order
    [D, idx] = sort(diag(e_raw), 'descend');
    eVecs = eVecs_raw(:, idx); % Reordering the eigenvectors to match sorted eigenvalues

    % Figuring out the percentage of impact
    sumD = sum(D);
    Percent_Impact = D / sumD;
    % Plotting
    figure;
    bar(Percent_Impact);
    %Some more formatting
    title('Explained Variance by Principal Components');
    xlabel('Principal Component');
    ylabel('Proportion of Variance Explained');
    %print('PCA_PLOT', '-dpng', '-r300');
 
    % Creating the feature matrix
    Feature = eVecs(:, 1:3);

end

function dispMuVar(mu,Var)

names = ["KO", "HD", "TSM", "WFC", "NVDA"];%Creating names for labeling
figure;
scatter(Var, mu, 'red');%Scatter plot of variances and return
%Formatting
xlabel("Variance");
ylabel("Expected Return")
text(Var, mu, names, 'Vert','bottom', 'Horiz','left', 'FontSize',7)
%print('Scatter', '-dpng', '-r300')
end
