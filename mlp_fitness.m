function [ y ] = mlp_fitness( x )
%mlp_fitness Summary of this function goes here
%   Detailed explanation goes here

global mlp_net net_in targets

mlp_net = setwb(mlp_net, x);
net_out = mlp_net(net_in);

mlp_mse = perform(mlp_net, targets, net_out);
mlp_regr = regression(targets, net_out, 'one');

y = mlp_mse - mlp_regr;

end

