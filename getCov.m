function [sigma, returns] = getCov(files)
f1 = files(1);
f2 = files(2);
f3 = files(3);
f4 = files(4);
f5 = files(5);

data1 = readmatrix(f1);
data2 = readmatrix(f2);
data3 = readmatrix(f3);
data4 = readmatrix(f4);
data5 = readmatrix(f5);

close_asset1 = data1(:,5);
close_asset2 = data2(:,5);
close_asset3 = data3(:,5);
close_asset4 = data4(:,5);
close_asset5 = data5(:,5);

tot_close = [close_asset1 close_asset2 close_asset3 close_asset4 close_asset5];

returns = diff(tot_close) ./ tot_close(1:end-1, :); % Percentage returns

sigma = cov(returns);

end