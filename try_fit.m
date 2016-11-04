function [ mse, regr ] = try_fit( num_hidden )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

global net_in
global targets

net = fitnet(num_hidden, 'trainlm');

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

[net, tr] = train(net, net_in, targets);

net_out = net(net_in);

errors = gsubtract(net_out, targets);

mse = perform(net, targets, net_out);
regr = regression(targets, net_out, 'one');

end