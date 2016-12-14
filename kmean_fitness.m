function [ y ] = kmean_fitness( x )
%kmean_feat Fitness function to compute the optimal number
% of clusters in k-mean clustering.

global net_in idx clust_c;

[idx, clust_c] = kmeans(net_in',x, 'Distance','cityblock');
silh = silhouette(net_in',idx,'cityblock');

y = -mean(silh);

end

