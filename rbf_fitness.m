function [ y ] = rbf_fitness( x )
%rbf_fitness Fitness function fot the GA on RBF network

global net_in
global targets

net = network(1,2,[1;1],[1;0],[0 0;1 0],[0 1]);
net.inputs{1}.size = 3;
net.layers{1}.size = 398;
net.inputWeights{1,1}.weightFcn = 'dist';
net.layers{1}.netInputFcn = 'netprod';
net.layers{1}.transferFcn = 'radbas';
net.layers{2}.size = 1;

hidden_bias(1:398) = x(1594);
wb = [x(1:1593)'; hidden_bias'; x(1595)];

net = setwb(net, wb);
net_out = net(net_in);

rbf_mse = perform(net, targets, net_out);
rbf_regr = regression(targets, net_out, 'one');

y = rbf_mse - rbf_regr;


end