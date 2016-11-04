function [ y ] = my_fitness( x )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

global features_norm
global targets

net = fitnet(20);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net_in = [features_norm(:,x(1)) features_norm(:,x(2)), ...
    features_norm(:,x(3))]';

[net, tr] = train(net, net_in, targets);

net_out = net(net_in);

% errors = gsubtract(net_out, targets);

perf = perform(net, targets, net_out);
regr = regression(targets, net_out, 'one');

y = perf - regr;

end

