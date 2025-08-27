function dispMuVar(mu, sigma);

n = length(mu);

for i = 1:n
    Variances(i) = sigma(i,i);
end
names = ["KO", "HD", "TSM", "WFC", "NVDA"];
figure;
scatter(Variances, mu, 'red');
xlabel("Variance");
ylabel("Expected Return")
text(Variances, mu, names, 'Vert','bottom', 'Horiz','left', 'FontSize',7)
%print('Scatter', '-dpng', '-r300')
end