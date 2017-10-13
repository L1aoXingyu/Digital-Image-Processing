I = imread('./lena.jpg');
h_I = compute_hist(I);
target = imread('./target.jpg');
h_target = compute_hist(target);
K = HistogramMatching(I, target);
h_K = compute_hist(K);

x = 0 : 255;
figure;
image(I);
title('origin image');

figure;
bar(x, h_I(:, 1));
title('origin image histogram');

figure;
image(target);
title('target image');

figure;
bar(x, h_target(:, 1));
title('target image histogram');

figure;
image(K);
title('transform image');

figure;
bar(x, h_K(:, 1));
title('transform image histogram');