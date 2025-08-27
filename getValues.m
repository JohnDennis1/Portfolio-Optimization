function [open_i, mean_close, RoR] = getValues(filename)

data = readmatrix(filename);

open_i = data(end, 2);

fin_close = data(1,5);

mean_close = mean(data(:,5));

RoR = (fin_close-open_i)/open_i;

end