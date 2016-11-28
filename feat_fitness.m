function [ y ] = feat_fitness( x )
%feat_fitness fitness function for the GA algorithm
%   The objective function is computed as the difference between
%   the input correlation and the target correlation for a set
%   of 3 possible inputs.

global feat_corr targ_corr alpha;

% This constant is needed to avoid the assignment of too much
% importance to the input correlation.
% corr_factor = 2;

% For each of the three inputs the correlation with respect to
% the other inputs is computed.
f1 = (sum(feat_corr(x(1),[x(2) x(3)])));% / corr_factor;
f2 = (sum(feat_corr(x(2),[x(1) x(3)])));% / corr_factor;
f3 = (sum(feat_corr(x(3),[x(1) x(2)])));% / corr_factor;

% Finally, for each input the correlation with respect to the target
% is computed.
t1 = abs(targ_corr(x(1)));
t2 = abs(targ_corr(x(2)));
t3 = abs(targ_corr(x(3)));

% Value returned by the fitness function.
y = alpha*(f1 + f2 + f3) - (1-alpha)*(t1 + t2 + t3); 

end

