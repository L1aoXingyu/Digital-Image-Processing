function h = compute_hist( img )
%compute histogram of an image

[m, n, c] = size(img);
h = zeros([256, c]);
for i = 1 : 256
    h(i, :) = sum(sum(img == (i - 1)));
end
end