function [ u, c ] = KMeans( x, k )
%Implement of Keans Algorithm for cluster colormap image
%   x is a n-dimentional data, and k means how many centers of the cluster
[m, n] = size(x);
u = zeros(k, n);
% initialize kmeans center using random uniform cutting min to max 
sort_x = sort(x, 1);
min_value = sort_x(1, :);
max_value = sort_x(m, :);
for i = 1 : k
    u(i, :) = min_value + (max_value - min_value) .* rand();
end

c = zeros([m, 1]);
last_error = 0;
% iter two formula to converge
while 1
    total_error = 0;
    % fisrt formula
    for i = 1 : m
        min_dist = sum((x(i, :) - u(1, :)) .^ 2);
        c(i) = 1;
        for j = 1 : k
            temp_dist = sum((x(i, :) - u(j, :)) .^ 2);
            if temp_dist < min_dist
                min_dist = temp_dist;
                c(i) = j;
            end
        end
        total_error = total_error + min_dist;
    end
    if abs(total_error - last_error) < 1
        break
    end
    last_error = total_error;
    % second formula
    for i = 1 : k
        temp_u = zeros([1, n]);
        count = 0;
        for j = 1 : m
            if c(j) == i
                temp_u = temp_u + x(j, :);
                count = count + 1;
            end
        end
        if count == 0
            u(i, :) = min_value + (max_value - min_value) .* rand();
        else
            u(i, :) = temp_u ./ count;
        end
    end
end

