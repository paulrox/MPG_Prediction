function [ y ] = rbf_fitness2( x )
%rbf2_fitness RBF network fitness function

global rbf_net2 net_in targets rbf_net2_IW rbf_net2_b clust_size;

LW = cell(2,2);
LW{2,1} = x(1:clust_size);
rbf_net2_b{2} = x(clust_size+1);
wb = formwb(rbf_net2, rbf_net2_b, rbf_net2_IW, LW);
rbf_net2 = setwb(rbf_net2, wb);

rbf_net_out = rbf_net2(net_in);

rbf_mse = perform(rbf_net2, targets, rbf_net_out);
rbf_regr = regression(targets, rbf_net_out, 'one');

y = rbf_mse - rbf_regr;

end

