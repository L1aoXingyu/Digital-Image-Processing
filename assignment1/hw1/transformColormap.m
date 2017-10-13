function [ colormap_img, new_cmap ] = transformColormap( img, colormap_dim )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

[height, width, channel] = size(img);
cmap = zeros(height * width, 3);
for i = 1 : height
    for j = 1 : width
        cmap((i - 1) * width + j, :) = double(img(i, j, :)) / 255; 
    end
end

[new_cmap, c] = KMeans(cmap, colormap_dim);

colormap_img = zeros([height, width]);
for i = 1 : height
    for j = 1 : width
        colormap_img(i, j) = c((i - 1) * width + j);
    end
end

end