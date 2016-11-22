function [ y ] = mlp_fitness( x )
%mlp_fitness Summary of this function goes here
%   Detailed explanation goes here

global mlp_net net_in targets

[iw, lw, b] = extractwb(x, mlp_net.layers{1}.size, mlp_net.inputs{1}.size);

IW = cell(2,1);
LW = cell(2,2);
B = cell(2,1);

IW{1} = iw;
LW{2,1} = lw;
B{1} = b(1:size(b,2)-1);
B{2} = b(size(b,2));

wb = formwb(mlp_net, B, IW, LW);
mlp_net = setwb(mlp_net, wb);
net_out = mlp_net(net_in);

mlp_mse = perform(mlp_net, targets, net_out);
mlp_regr = regression(targets, net_out, 'one');

y = mlp_mse - mlp_regr;

end

