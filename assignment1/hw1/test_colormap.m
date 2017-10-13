clear all;
close all;
clc;

img = imread('./jim.jpg');
figure;
image(img);
title('true image');

[colormap_img, cmap] = transformColormap(img, 256);

figure;
image(colormap_img);
colormap(cmap);
title('colormap image');

imwrite(colormap_img, cmap, 'hw1_colormap.bmp', 'bmp');

[img1, cmap1] = imread('./hw1_colormap.bmp', 'bmp');
trans_true_img = transformTrueimage(img1, cmap1);

% test error between real image and color map image
[h, w, channel] = size(trans_true_img);
error = 0;
for i = 1 : h
    for j = 1 : w
        error = error + sum(abs(img(i, j, :) - trans_true_img(i, j, :)));
    end
end
error = error / (h * w * channel);
fprintf('error is %.4f\n', error);