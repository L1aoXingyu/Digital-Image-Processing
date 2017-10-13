function [real_img ] = transformTrueimage(img, cmap)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[height, width] = size(img);
real_img = zeros(height, width, 3);
for i = 1:height
    for j = 1:width
        idx = img(i, j) + 1;
        real_color = cmap(idx, :) * 255.0;
        real_img(i, j, :) = real_color;
    end
end
real_img = uint8(real_img);
end
