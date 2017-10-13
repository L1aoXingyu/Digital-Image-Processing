function [ h, lut, eq_I] = HistogramEqualization( I )
%HistogramEqualization
%   compute histogram of an image and do histogram equalization
[m, n, c] = size(I);
h = compute_hist(I);
[m_h, n_h] = size(h);
total = sum(h);

% construct lut
lut = zeros([m_h, n_h]);
lut(1, :) = h(1, :) ./ total;
for i = 2 : m_h
    lut(i, :) = lut(i - 1, :) + h(i, :) ./ total;
end
for i = 1 : m_h
    lut(i, :) = round(lut(i, :) .* 255);
end
eq_I = zeros(size(I));
for i = 1 : m
    for j = 1 : n
        for k = 1 : c
            eq_I(i, j, k) = lut(I(i, j, k) + 1, k);
        end
    end
end
eq_I = uint8(eq_I);
end

