function [ y ] = mlp_fitness2( x )
%mlp_fitness2 Summary of this function goes here
%   Detailed explanation goes here

global mlp_net2 net_in targets

mlp_net2 = setwb(mlp_net2, x);
net_out = mlp_net2(net_in);

mlp_mse = perform(mlp_net2, targets, net_out);
mlp_regr = regression(targets, net_out, 'one');

y = mlp_mse - mlp_regr;


end

