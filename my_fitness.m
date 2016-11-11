function [ y ] = my_fitness( x )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

global net_in
global targets

net = feedforwardnet(10);

net = configure(net, net_in, targets);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%view(net);
net = setwb(net, x');
net_out = net(net_in);

% errors = gsubtract(net_out, targets);

perf = perform(net, targets, net_out);
regr = regression(targets, net_out, 'one');

y = perf - regr;

end

