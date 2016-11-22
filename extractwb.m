function [ IW, LW, b ] = extractwb( wb, numHidden, numInput )
%extractwb Extracts the weights and biases from an array composed
% by 'compresswb'.

IW = zeros(numHidden, numInput);

for i=1:numHidden*numInput
    IW(ceil(i/numInput), mod(i, numInput)+1) = wb(i);
end;

LW = wb(numHidden*numInput+1:(numHidden*numInput)+numHidden);

b = wb((numHidden*numInput)+numHidden+1:size(wb, 2));

end

