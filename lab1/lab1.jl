using Pkg

using Images
using FileIO
using ImageView
using ImageFiltering
using Statistics

# 1

org_img = load("glupi_kot.jpg")

# 2

function to_grey(img)
    new_img = channelview(img)
    new_img = mean(new_img, dims=1)
    new_img = dropdims(new_img, dims=1)
    return new_img
end

grey_img = to_grey(org_img)

imshow(grey_img)

# size(org_img) == size(grey_img)

# 3

function custom_conv(img, step, func)
    h, w = size(img)
    polled_img = zeros(h÷step, w÷step)
    for i ∈ 1:step:(h-step+1)
        for j ∈ 1:step:(w-step+1)
            polled_img[i÷step+1, j÷step+1] = func(img[i:i+step-1, j:j+step-1])
        end
    end
    return polled_img 
end

function max_pooling(img)
    custom_max = x -> max(x...)
    return custom_conv(img, 4, custom_max)
end

function avg_pooling(img)
    return custom_conv(img, 4, mean)
end

ap = avg_pooling(grey_img)
mp = max_pooling(grey_img)

imshow(ap, name="avg_pooling")
imshow(mp, name="max_pooling")

# 4

function gaussian_blur(img, sigma, n)
    return imfilter(img, Kernel.gaussian((sigma, sigma), (n, n)))
end

img_gaus = gaussian_blur(ap, 0.7, 7)
imshow(img_gaus)

# 5

hor_sobel = [[1, 2, 1] [0, 0, 0] [-1, -2, -1]]
ver_sobel = [[1, 0, -1] [2, 0, -2] [1, 0, -1]]

function conv_with_matrix(img, mat)
    return imfilter(img, mat)
end


img_sobel_hor = conv_with_matrix(img_gaus, hor_sobel)
img_sobel_ver = conv_with_matrix(img_gaus, ver_sobel)

imshow(img_sobel_hor, name="Horizontal")
imshow(img_sobel_ver, name="Vertical")

function merge_sobels!(img1, img2)
    img1 .^= 2
    img2 .^= 2
    img1 .+= img2
    return .√(img1)
end

img_sobel = merge_sobels!(copy(img_sobel_hor), copy(img_sobel_ver))
img_angle = atand.(img_sobel_ver, img_sobel_hor)

imshow(img_sobel)

# 6

function non_max_suppression(img, angles)
    h, w = size(img)
    Z = zeros(h, w)
    angles[angles .< 0] .+= 180

    for i ∈ 2:h-1
        for j ∈ 2:w-1
            q = 255
            r = 255
            if (0 <= angles[i,j] < 22.5) || (157.5 <= angles[i,j] <= 180)
                q = img[i, j+1]
                r = img[i, j-1]
            elseif (22.5 <= angles[i,j] < 67.5)
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            elseif (67.5 <= angles[i,j] < 112.5)
                q = img[i+1, j]
                r = img[i-1, j]
            elseif (112.5 <= angles[i,j] < 157.5)
                q = img[i-1, j-1]
                r = img[i+1, j+1]
            end

            if (img[i,j] >= q) && (img[i,j] >= r)
                Z[i,j] = img[i,j]
            else
                Z[i,j] = 0
            end
        end
    end
    return Z
end

img_sup = non_max_suppression(img_sobel, img_angle)
imshow(img_sup)

# 7

function edge_filter!(img, λ)
    # angles[angles .< 0] .+= 180
    img[img .< λ] .= 0
    img[img .>= λ] .= 1
    return img
end

img_filtered = edge_filter!(copy(img_sup), 0.6)
imshow(img_filtered)

# 9

img_resized = imresize(img_filtered, size(org_img))
imshow(img_resized)

function add_images_with_green(normal_img, bw_matrix)
    result_img = copy(normal_img)
    for x in CartesianIndices(bw_matrix)   
        if bw_matrix[x] > 0.5    
            result_img[x] = RGB(0.0, 1.0, 0.0)
        else
            result_img[x] = normal_img[x]  
        end
    end
    return result_img
end

img_final = add_images_with_green(org_img, img_resized)
imshow(img_final)

function find_edges!(img)
    org_img = copy(img)
    img = to_grey(img)
    imshow(img, name="grey")
    img = avg_pooling(img)
    # imshow(img, name="avg pooling")
    img = gaussian_blur(img, 2, 3)
    imshow(img)

    img_sobel_hor = conv_with_matrix(img, hor_sobel)
    img_sobel_ver = conv_with_matrix(img, ver_sobel)

    img_sobel = merge_sobels!(copy(img_sobel_hor), copy(img_sobel_ver))
    imshow(img_sobel, name="full sobel")
    # img_angle = atand.(img_sobel_hor, img_sobel_ver)
    img_angle = atand.(img_sobel_ver, img_sobel_hor)

    img = non_max_suppression(img_sobel, img_angle)
    imshow(img, name="non max suppression")
    img = edge_filter!(copy(img), 0.4)
    imshow(img, name="edge filter")
    img = imresize(img, size(org_img))
    imshow(img, name="resize")

    img = add_images_with_green(org_img, img)
    imshow(img, name="final")
end

mewa = load("mewa.png")
floppa = load("floppa.jpg")

find_edges!(copy(mewa))