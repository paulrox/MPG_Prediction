function [ y ] = rbf_fitness( x )
%rbf_fitness Fitness function fot the GA on RBF network.

% We store weights and biases in an array structured
% like this: [InputLayerWeights HiddenLayeWeights 
% HiddenLayerBias OutputLayerBias].

global rbf_net net_in targets

IW = cell(2,1);
LW = cell(2,2);
B = cell(2,1);

[iw, lw, b] = extractwb(x, rbf_net.layers{1}.dimensions, ...
    rbf_net.inputs{1}.size);
extend_hb = zeros(rbf_net.layers{1}.dimensions,1);
extend_hb(1:size(extend_hb,1)) = b(1);

IW{1} = iw;
LW{2,1} = lw;
B{1} = extend_hb;
B{2} = b(2);

wb = formwb(rbf_net, B, IW, LW);

% hidden_bias(1:398) = x(1594);
% wb = [x(1:1593)'; hidden_bias'; x(1595)];

rbf_net = setwb(rbf_net, wb);
rbf_net_out = rbf_net(net_in);

rbf_mse = perform(rbf_net, targets, rbf_net_out);
rbf_regr = regression(targets, rbf_net_out, 'one');

y = rbf_mse - rbf_regr;


end