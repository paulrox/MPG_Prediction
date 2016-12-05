%% Main script.
clearvars; clc; close all;

global net_in targets features_norm feat_corr targ_corr;
global history;
% features coloumns indices
cyl_col = 1;
disp_col = 2;
hp_col = 3;
wgt_col = 4;
acc_col = 5;
year_col = 6;
orig_col = 7;
name_col = 8;

history = struct;

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

%% Normalize the targets
target_m = mean(mpg);
target_d = std(mpg);

target_norm = (mpg - target_m) / target_d;

%% Extract the correlation matrices and find the best features set

% Correlation between the input features.
feat_corr = corr(features_norm);

% Correlation between input features and targets.
targ_corr = corr(features_norm, target_norm);

% Setup the GA to find the set of features which maximizes the output
% correlation and minimizes the input correlation. In particular we want
% to solve a linear multiobjective problem where we want to minimize the
% input correlation and to maximize the output correlation. We apply the
% scalarization method to obtain the set of Pareto minimum points.
% fitnessFcn = @feat_fitness;
% nvar = 3;
% 
% options = gaoptimset;
% options = gaoptimset(options,'TolFun', 1e-8, 'Generations', 300);
% 
% global alpha;
% 
% feat_sol = [];
% 
% for alpha=1:-0.005:0
%     [x, fval] = ga(fitnessFcn, nvar, [], [], [], [], [1; 1; 1], [7; 7; 7], ...
%         [], [1 2 3], options);
%     feat_sol = [feat_sol; sort(x)];
% end;

% By observing the solutions, there are 3 possible sets of features:
% [2    5   7]
% [2    6   7]
% [2    4   7]
% And by comparing the resulting MSE and R-Value in a MLP network we
% observed that the features set [2 6 7] gives the best results.


%% Setup the NN inputs and targets

% Select 3 features
net_in = [features_norm(:,2) features_norm(:,6) features_norm(:,7)]';

targets = target_norm';

%% Multi-Layer Perceptron (1 LAYER)
%We use the GA to find the best weights and biases. 

global mlp_net;

mlpFitness = @mlp_fitness;

mlp_nets = cell(10, 2);

for i=5:15
    
    mlp_net = feedforwardnet(i);
    mlp_net = configure(mlp_net, net_in, targets);
    %mlp_net.inputs{1}.processFcns = {};
    mlp_net.divideParam.trainRatio = 70/100;
    mlp_net.divideParam.valRatio = 15/100;
    mlp_net.divideParam.testRatio = 15/100;
    
    mlp_nvar = mlp_net.numWeightElements;

    % Initial set of weights, computed by Matlab.
    
    mlp_trained = train(mlp_net, net_in, targets);
    trained_wb = getwb(mlp_trained);

    mlp_options = gaoptimset;
    mlp_options = gaoptimset(mlp_options,'TolFun', 1e-8, 'Display', 'iter', ...
        'SelectionFcn', @selectionroulette, ...
        'CrossoverFcn', @crossoversinglepoint, ...
        'MutationFcn', @mutationgaussian, ...
        'Generations', 3, ...
        'InitialPopulation', trained_wb', ...
        'OutputFcn', @ga_output);
        %'PlotFcns', @gaplotbestf);

  [mlp_weights, mlp_fval, ~, mlp_output] = ga(mlpFitness, mlp_nvar, ...
       [], [], [], [], [], [], [], [], mlp_options);
    
    mlp_nets{i-4, 1} = mlp_net;
    mlp_nets{i-4, 1} = setwb(mlp_nets{i-4, 1}, mlp_weights);
    mlp_nets{i-4, 2} = history;
end;

%% Multi-Layer Perceptron (2 LAYERS)

global mlp_net2

mlpFitness2 = @mlp_fitness2;

mlp_nets2 = cell(10, 10);

for i=5:15
    for j=5:15
    
    mlp_net2 = feedforwardnet([i, j]);
    mlp_net2 = configure(mlp_net2, net_in, targets);
    mlp_net2.divideParam.trainRatio = 70/100;
    mlp_net2.divideParam.valRatio = 15/100;
    mlp_net2.divideParam.testRatio = 15/100;
    
    mlp_nvar2 = mlp_net2.numWeightElements;

    % Initial set of weights, computed by Matlab.
    
    mlp_trained2 = train(mlp_net2, net_in, targets);
    trained_wb2 = getwb(mlp_trained2);

    mlp_options2 = gaoptimset;
    mlp_options2 = gaoptimset(mlp_options2,'TolFun', 1e-8, 'Display', 'iter', ...
        'SelectionFcn', @selectionroulette, ...
        'CrossoverFcn', @crossoversinglepoint, ...
        'MutationFcn', @mutationgaussian, ...
        'Generations', 300, ...
        'InitialPopulation', trained_wb2', ...
        'OutputFcn', @ga_output); 
        %'PlotFcns', @gaplotbestf);

   [mlp_weights2, mlp_fval2, ~, mlp_output2] = ga(mlpFitness2, mlp_nvar2, [], [], [], [], ...
       [], [], [], [], mlp_options2);
    
    mlp_nets2{i-4, j-4}{1} = mlp_net2;
    mlp_nets2{i-4, j-4}{1} = setwb(mlp_nets2{i-4, j-4}{1}, mlp_weights2);
    mlp_nets2{i-4, j-4}{2} = history;
    end;
end;

%% Radial Basis Function Network (Max hidden neurons)
% With the GA we want to find the best spread and centers for the
% RBF neurons.

% Create the RBF network.
global rbf_net
rbf_net = network(1,2,[1;1],[1;0],[0 0;1 0],[0 1]);
rbf_net.inputs{1}.size = 3;
rbf_net.layers{1}.size = 398;
rbf_net.inputWeights{1,1}.weightFcn = 'dist';
rbf_net.layers{1}.netInputFcn = 'netprod';
rbf_net.layers{1}.transferFcn = 'radbas';
rbf_net.layers{2}.size = 1;

load('rbf_best_wb.mat');
rbf_init_b = cell(2, 1);
rbf_init_b{1} = brbf_b{1}(1);
rbf_init_b{2} = brbf_b{2};
rbf_init_wb = compresswb(brbf_IW, brbf_LW, rbf_init_b);

rbfFitness = @rbf_fitness;
rbf_nvar = 1594;

rbf_options = gaoptimset;
rbf_options = gaoptimset(rbf_options,'TolFun', 1e-8, 'Display', 'iter', ...
    'SelectionFcn', @selectionroulette, ...
    'CrossoverFcn', @crossoversinglepoint, ...
    'MutationFcn', @mutationadaptfeasible, ...
    'Generations', 150, ...
    'CreationFcn', @gacreationlinearfeasible, ...
    'InitialPopulation', rbf_init_wb, ...
    'OutputFcn', @ga_output);
    %'PlotFcns', @gaplotbestf,

[rbf_weights, rbf_fval, ~, rbf_output] = ga(rbfFitness, rbf_nvar, [zeros(1592, 1594); ...
    zeros(1, 1592) -1 0; zeros(1, 1594)], [zeros(1592,1); 0.001; 0], ...
    [], [], [], [], [], [], rbf_options);

IW_rbf = cell(2,1);
LW_rbf = cell(2,2);
B_rbf = cell(2,1);

[iw_rbf, lw_rbf, b_rbf] = extractwb(rbf_weights, ...
    rbf_net.layers{1}.dimensions, rbf_net.inputs{1}.size);
extend_hb = zeros(rbf_net.layers{1}.dimensions,1);
extend_hb(1:size(extend_hb,1)) = b_rbf(1);

IW_rbf{1} = iw_rbf;
LW_rbf{2,1} = lw_rbf;
B_rbf{1} = extend_hb;
B_rbf{2} = b_rbf(2);

wb = formwb(rbf_net, B_rbf, IW_rbf, LW_rbf);

rbf_net = setwb(rbf_net, wb);

rbf_history = history;

%% Radial Basis Function Network (Best number of hidden neurons)
% First of all we use tje k-means clustering to find clusters of input data
% to be used with the RBF network.

global clust_c;

kmeanFitness = @kmean_fitness;
nvar = 1;

kmean_options = gaoptimset;
kmean_options = gaoptimset(kmean_options,'TolFun', 1e-8, ...
    'Generations', 10, 'Display', 'iter', 'FitnessLimit', -1);

global rbf_net2 rbf_net2_IW rbf_net2_b clust_size

rbf_nets = cell(10, 2);

for i=5:15
    clust_size = i;

    [idx, clust_c] = kmeans(net_in', clust_size, 'Distance','cityblock');

    % Compute the maximum distance among centroids.
    max_dist = max(pdist(clust_c));

    % Compute the spread.
    spread = max_dist/(sqrt(clust_size));

    % Find the output layer weights and bias using GA.
    rbf_net2 = network(1,2,[1;1],[1;0],[0 0;1 0],[0 1]);
    rbf_net2.inputs{1}.size = 3;
    rbf_net2.layers{1}.size = clust_size;
    rbf_net2.inputWeights{1,1}.weightFcn = 'dist';
    rbf_net2.layers{1}.netInputFcn = 'netprod';
    rbf_net2.layers{1}.transferFcn = 'radbas';
    rbf_net2.layers{2}.size = 1;

    rbf_net2_IW = cell(2, 1);
    rbf_net2_IW{1} = clust_c;
    hidden_b(1:clust_size) = spread;
    rbf_net2_b = cell(2, 1);
    rbf_net2_b{1} = hidden_b;

    rbfFitness2 = @rbf_fitness2;
    rbf_nvar2 = clust_size+1;

    rbf_options2 = gaoptimset;
    rbf_options2 = gaoptimset(rbf_options2,'TolFun', 1e-8, 'Display', 'iter', ...
        'SelectionFcn', @selectionroulette, ...
        'CrossoverFcn', @crossoversinglepoint, ...
        'MutationFcn', @mutationadaptfeasible, ...
        'Generations', 300, ...
        'OutputFcn', @ga_output);
        %'PlotFcns', @gaplotbestf

    [rbf_weights2, rbf_fval2, ~, rbf_output2] = ga(rbfFitness2, rbf_nvar2, [], [], ...
        [], [], [], [], [], [], rbf_options2);

    LW = cell(2,2);
    LW{2,1} = rbf_weights2(1:clust_size);

    rbf_net2_b{2} = rbf_weights2(clust_size+1);
    wb = formwb(rbf_net2, rbf_net2_b, rbf_net2_IW, LW);
    rbf_net2 = setwb(rbf_net2, wb);
    
    rbf_nets{i-4, 1} = rbf_net2;
    rbf_nets{i-4, 2} = history;
end;