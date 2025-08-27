function [eVecs, eVecs_raw, D, Percent_Impact, Feature, Var] = getPCA(sigma)
    % Compute eigenvalues and eigenvectors
    [eVecs_raw, e_raw] = eig(sigma);
    Var = diag(sigma); % Original variances (diagonal elements of sigma)

    % Extract and sort eigenvalues in descending order
    [D, idx] = sort(diag(e_raw), 'descend');
    eVecs = eVecs_raw(:, idx); % Reorder eigenvectors to match sorted eigenvalues

    % Calculate the proportion of variance explained
    sumD = sum(D);
    Percent_Impact = D / sumD;
    % Plot explained variance
    figure;
    bar(Percent_Impact);
    title('Explained Variance by Principal Components');
    xlabel('Principal Component');
    ylabel('Proportion of Variance Explained');
    print('PCA_PLOT', '-dpng', '-r300');
    

    % Select the top 3 principal components
    Feature = eVecs(:, 1:3); % Use top 3 eigenvectors as features

end