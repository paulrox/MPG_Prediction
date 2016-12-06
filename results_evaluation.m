%% Clear all and load the data.
clearvars; clc; close all;

load('results.mat', 'mlp_nets', 'mlp_nets2', 'rbf_net', 'rbf_nets', ...
    'rbf_history');

cyl_col = 1;
disp_col = 2;
hp_col = 3;
wgt_col = 4;
acc_col = 5;
year_col = 6;
orig_col = 7;
name_col = 8;

%% Extract all the features except the car names.
extract_allfeatures;

% Input features matrix sizes
f_col = size(features,2);
f_row = size(features,1);

%% Extract the MPG.
extract_mpg;

%% Handle NaNs
% We substitute all the NaNs with the mean value of that feature.
% We already know that there are 8 missing values in the horsepower
% column.

% Compute the mean of the numeric values.
notNaN = features(~isnan(features(:,3)),hp_col);
notNaN_mean = mean(notNaN);

% Substitute all the NaNs with the mean value.
features(isnan(features(:,3)), hp_col) = notNaN_mean;

%% Normalize the features
% For each feature we compute the mean value and the standard
% deviation.
feat_m = zeros(1, f_col);
feat_d = zeros(1, f_col);

for i=1:f_col
    feat_m(i) = mean(features(:,i));
    feat_d(i) = std(features(:,i));
end;

% Now subtract the mean value from each feature value and divide
% by its standard deviation.
features_norm = zeros(size(features,1),size(features,2));
for i=1:f_col
    features_norm(:,i) = (features(:,i) - feat_m(i)) / feat_d(i);
end;

%% Setup the NN inputs and targets
% Select 3 features
net_in = [features_norm(:,2) features_norm(:,6) features_norm(:,7)]';
targets = mpg';

%% Multi Layer Perceptron Comparison (1 HIDDEN LAYER)
% We compare the mse values and r-values of the networks to see which one
% is the best.
perf_mlp1 = zeros(11,3);

for i=1:11
    out = mlp_nets{i,1}(net_in);
    out = out*std(targets) + mean(targets);
    perf_mlp1(i,1) = 4 + i;
    perf_mlp1(i,2) = perform(mlp_nets{i,1},targets,out);
    perf_mlp1(i,3) = regression(targets, out,'one');
end;

% Plot all the results
figure;
bar(perf_mlp1(:,1),perf_mlp1(:,2))
xlabel('Hidden Neurons');
ylabel('MSE');
title('MLP - 1 Hidden Layer - Comparison');

% Find the best
mlp1_best = perf_mlp1(perf_mlp1(:,2) == min(perf_mlp1(:,2)));

% Extract data for the best architecture
figure;
out = mlp_nets{mlp1_best-4,1}(net_in);
out = out*std(targets) + mean(targets);
err = targets - out;

ploterrhist(err);
title('MLP - 1 Hidden Layer - 13 Hidden Neurons');

% GA plots
figure;
scatter(mlp_nets{mlp1_best-4,2}.gen,mlp_nets{mlp1_best-4,2}.best);
hold on;
scatter(mlp_nets{mlp1_best-4,2}.gen,mlp_nets{mlp1_best-4,2}.mean);
xlabel('Generations');
ylabel('Score');
title('MLP - 1 Hidden Layer - Best GA');
legend('Best','Mean');



%% Multi Layer Perceptron Comparison (2 HIDDEN LAYERS)
% We compare the mse values and r-values of the networks to see which one
% is the best.
perf_mlp2 = zeros(100,4);
y_bar = zeros(11,11);

for i=1:11
    for j=1:11
        out = mlp_nets2{i,j}{1}(net_in);
        out = out*std(targets) + mean(targets);
        perf_mlp2(j+((i-1)*11),1) = 4 + i;
        perf_mlp2(j+((i-1)*11),2) = 4 + j;
        perf_mlp2(j+((i-1)*11),3) = perform(mlp_nets2{i,j}{1},targets,out);
        perf_mlp2(j+((i-1)*11),4) = regression(targets,out,'one');
        y_bar(i,j) = perf_mlp2(j+((i-1)*11),3);
    end;
end;

% Plot all the results
figure;
bar(perf_mlp2(1:11,2),y_bar);
legend('5 HN(Layer 2)', '6 HN(Layer 2)', '7 HN(Layer 2)', '8 HN(Layer 2)', ...
    '9 HN(Layer 2)', '10 HN(Layer 2)', '11 HN(Layer 2)', '12 HN(Layer 2)', ...
    '13 HN(Layer 2)', '14 HN(Layer 2)', '15 HN(Layer 2)');
xlabel('Hidden Neurons(Layer 1)');
ylabel('MSE');
title('MLP - 2 Hidden Layers - Comparison');

% Find the best
mlp2_best = perf_mlp2(perf_mlp2(:,3) == min(perf_mlp2(:,3)),:);

% Extract data for the best architecture
figure;
out = mlp_nets2{mlp2_best(1)-4,mlp2_best(2)-4}{1}(net_in);
out = out*std(targets) + mean(targets);
err = targets - out;

ploterrhist(err);
title('MLP - 2 Hidden Layers - [13 15] Hidden Neurons');

% GA plots
figure;
scatter(mlp_nets2{mlp2_best(1)-4,mlp2_best(2)-4}{2}.gen, ...
    mlp_nets2{mlp2_best(1)-4,mlp2_best(2)-4}{2}.best);
hold on;
scatter(mlp_nets2{mlp2_best(1)-4,mlp2_best(2)-4}{2}.gen, ...
    mlp_nets2{mlp2_best(1)-4,mlp2_best(2)-4}{2}.mean);
xlabel('Generations');
ylabel('Score');
title('MLP - 2 Hidden Layer - Best GA');
legend('Best','Mean');


%% Radial Basis Function Network Comparison (MAX NEURONS)
% We evaluate the performance od a RBF network with the maximum number of
% hidden neurons.
perf_rbf = zeros(1, 2);

out = rbf_net(net_in);
out = out*std(targets) + mean(targets);
perf_rbf(1) = perform(rbf_net,targets,out);
perf_rbf(2) = regression(targets, out, 'one');

figure;
err = targets - out;
ploterrhist(err);
title('RBF - Max Hidden Neurons');

% GA plots
figure;
scatter(rbf_history.gen,rbf_history.best);
hold on;
scatter(rbf_history.gen,rbf_history.mean);
xlabel('Generations');
ylabel('Score');
title('RBF - Max Hidden Neurons - Best GA');
legend('Best','Mean');


%% Radial Basis Function Network Comparison (VARIABLE NEURONS)
% We evaluate the performance of a RBF network with a variable number of
% hidden neuros which centers are chosen using a clustering algorithm.
perf_rbf2 = zeros(11, 3);

for i=1:11
    out = rbf_nets{i,1}(net_in);
    out = out*std(targets) + mean(targets);
    perf_rbf2(i,1) = i + 4;
    perf_rbf2(i,2) = perform(rbf_nets{i,1},targets,out);
    perf_rbf2(i,3) = regression(targets,out,'one');
end;

% Plot all the results
figure;
bar(perf_rbf2(:,1),perf_rbf2(:,2))
xlabel('Hidden Neurons');
ylabel('MSE');
title('RBF with Clustering - Comparison');

% Find the best
rbf2_best = perf_rbf2(perf_rbf2(:,2) == min(perf_rbf2(:,2)));
[idx, clust_c] = kmeans(net_in', rbf2_best, 'Distance','cityblock');

% Plot clustered data
figure;
symbols = ['o' '+' '*'];
h = zeros(1,rbf2_best);
for i = 1:rbf2_best
    clust = find(idx==i);
    h(i) = scatter3(net_in(1,clust),net_in(2,clust),net_in(3,clust),symbols(mod(i,3)+1));
    hold on
    scatter3(clust_c(i,1),clust_c(i,2),clust_c(i,3),150,'kx');
    
end
xlabel('Displacement');
ylabel('Model Year');
zlabel('Origin');
title('Observation partition in 14 clusters');
grid on
hold off

%Plot cluster silhouette
figure;
silhouette(net_in',idx, 'cityblock');

% Extract data for the best architecture
figure;
out = rbf_nets{rbf2_best-4,1}(net_in);
out = out*std2(targets) + mean(targets);
err = targets - out;

ploterrhist(err);

% GA plots
figure;
scatter(rbf_nets{rbf2_best-4,2}.gen,rbf_nets{rbf2_best-4,2}.best);
hold on;
scatter(rbf_nets{rbf2_best-4,2}.gen,rbf_nets{rbf2_best-4,2}.mean);
xlabel('Generations');
ylabel('Score');
title('RBF - Max Hidden Neurons - Best GA');
legend('Best','Mean');