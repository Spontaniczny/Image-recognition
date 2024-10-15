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

function max_pooling(img, n=4)
    custom_max = x -> max(x...)
    return custom_conv(img, n, custom_max)
end

function avg_pooling(img, n=4)
    return custom_conv(img, n, mean)
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

ver_sobel = [[1, 2, 1] [0, 0, 0] [-1, -2, -1]]
hor_sobel = [[1, 0, -1] [2, 0, -2] [1, 0, -1]]

function conv_with_matrix(img, mat)
    return imfilter(img, mat)
end


img_sobel_ver = conv_with_matrix(img_gaus, ver_sobel)
img_sobel_hor = conv_with_matrix(img_gaus, hor_sobel)

imshow(img_sobel_hor, name="Horizontal")
imshow(img_sobel_ver, name="Vertical")

function merge_sobels!(img1, img2)
    img1 .^= 2
    img2 .^= 2
    img1 .+= img2
    return .√(img1)
end

img_sobel = merge_sobels!(copy(img_sobel_hor), copy(img_sobel_ver))
img_angle = atand.(img_sobel_hor, img_sobel_ver)
img_angle .*= -1

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

function find_edges!(img, debug=false)
    org_img = copy(img)
    img = to_grey(img)
    if debug imshow(img, name="grey") end

    img = avg_pooling(img, 2)
    imshow(img, name="avg pooling")

    img = gaussian_blur(img, 1.4, 7)
    if debug imshow(img, name="gaussian_blur") end

    img_sobel_hor = conv_with_matrix(img, hor_sobel)
    img_sobel_ver = conv_with_matrix(img, ver_sobel)
 
    img_sobel = merge_sobels!(copy(img_sobel_hor), copy(img_sobel_ver))
    if debug imshow(img_sobel, name="full sobel") end
    
    img_angle = atand.(img_sobel_hor, img_sobel_ver)
    img_angle .*= -1
    img = non_max_suppression(img_sobel, img_angle)
    if debug imshow(img, name="non max suppression") end

    img = edge_filter!(copy(img), 0.1) 
    if debug imshow(img, name="edge filter") end

    img = imresize(img, size(org_img))
    if debug imshow(img, name="resize") end
    
    img = add_images_with_green(org_img, img)
    imshow(img, name="final")
end

function find_edges_and_save!(img, debug=false)
    org_img = copy(img)
    img = to_grey(img)
    if debug imshow(img, name="grey") end
    save("1_grey.png", img)
    save("2_max_pooling.png", max_pooling(img, 2))
    img = avg_pooling(img, 2)
    imshow(img, name="avg pooling")
    save("2_avg_pooling.png", img)

    img = gaussian_blur(img, 1.4, 7)
    if debug imshow(img, name="gaussian_blur") end
    save("3_gaussian_blur.png", img)
    img_sobel_hor = conv_with_matrix(img, hor_sobel)
    img_sobel_ver = conv_with_matrix(img, ver_sobel)
 
    save("4_sobel_hor.png", (img_sobel_hor .- minimum(img_sobel_hor)) ./ (maximum(img_sobel_hor) - minimum(img_sobel_hor)))
    save("4_sobel_ver.png", (img_sobel_ver .- minimum(img_sobel_ver)) ./ (maximum(img_sobel_ver) - minimum(img_sobel_ver)))
    img_sobel = merge_sobels!(copy(img_sobel_hor), copy(img_sobel_ver))
    if debug imshow(img_sobel, name="full sobel") end
    save("5_full_sobel.png", (img_sobel .- minimum(img_sobel)) ./ (maximum(img_sobel) - minimum(img_sobel)))
    
    img_angle = atand.(img_sobel_hor, img_sobel_ver)
    img_angle .*= -1
    save("5_angles.png", (img_angle .- minimum(img_angle)) ./ (maximum(img_angle) - minimum(img_angle)))
    img = non_max_suppression(img_sobel, img_angle)
    if debug imshow(img, name="non max suppression") end
    save("6_non_max_suppression.png", (img .- minimum(img)) ./ (maximum(img) - minimum(img)))
    img = edge_filter!(copy(img), 0.1) 
    if debug imshow(img, name="edge filter") end
    save("7_edge_filter.png", img)
    img = imresize(img, size(org_img))
    if debug imshow(img, name="resize") end
    save("8_rezised_to_org_size.png", img)
    img = add_images_with_green(org_img, img)
    imshow(img, name="final")
    save("9_final_img.png", img)
end

mewa = load("mewa.png")
floppa = load("floppa.jpg")


find_edges!(copy(floppa), false)