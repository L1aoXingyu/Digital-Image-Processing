function [ K ] = HistogramMatching( I, target )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here
I_h = compute_hist(I);
total_I = sum(I_h);
target_h = compute_hist(target);
total_target = sum(target_h);
[m, n] = size(I_h);
% define pdf for image I
pdf_h = zeros([m, n]);
pdf_h(1, :) = I_h(1, :) ./ total_I;
for i = 2 : m
    pdf_h(i, :) = pdf_h(i - 1, :) + I_h(i, :) ./ total_I;
end
% define pdf for image target 
pdf_target = zeros([m, n]);
pdf_target(1, :) = target_h(1, :) ./ total_I;
for i = 2 : m
    pdf_target(i, :) = pdf_target(i - 1, :) + target_h(i, :) ./ total_target;
end
%define LUT for image transform with target
LUT = zeros([m, n]);
for i = 1 : 256
    for k = 1 : 3
        temp = pdf_h(i, k);
        for j = 1 : 256
            if pdf_target(j, k) > temp
                LUT(i, k) = j;
                break;
            end
        end
    end
end
% get new image close to target image
K = zeros(size(I));
[r, g, b] = size(K);
for i = 1 : r
    for j = 1 : g
        for c = 1 : b
            K(i, j, c) = LUT(I(i, j, c) + 1, c);
        end
    end
end
K = uint8(K);
end

