function [ wb ] = compresswb( IW, LW, b )
%COMPRESSWB Compress weights and biases in an array
%   Input layer weights are arranged by rows

size_IW = size(IW{1},1) * size(IW{1},2);
% rows_IW = size(IW{1},1);
col_IW = size(IW{1},2);

temp = zeros(1, size_IW);

for i=1:size_IW
    temp(i) = IW{1}(ceil(i/col_IW), mod(i,col_IW)+1);
end;

wb = [temp, LW{2,1}, b{1}', b{2}];




end

