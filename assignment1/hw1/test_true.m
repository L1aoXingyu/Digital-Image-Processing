clear;
clc;

[cimg, cmap] = imread('./hw1.bmp', 'bmp');
true_img = transformTrueimage(cimg, cmap);
imwrite(true_img, 'hw1_treu.jpg');

figure;
image(cimg);
colormap(cmap);
title('color map image');

figure;
image(true_img);
title('true image');
