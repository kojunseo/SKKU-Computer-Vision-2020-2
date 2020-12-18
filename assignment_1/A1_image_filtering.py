import cv2
import numpy as np
import math 
import time
import os
from about_images import show_and_save, load_lenna, load_shape, show



def _padding(img, size):
    img_up = np.insert(img[0,:],0 ,[img[0,0]]*size[0][0])
    img_up = np.append(img_up, [img[0,-1]]*size[0][1])
    img_down = np.insert(img[-1,:],0, [img[-1,0]]*size[0][0])
    img_down = np.append(img_down, [img[-1,-1]]*size[0][1])
    
    img_left = img[:,0].reshape((img.shape[0],1))
    img_right = img[:,-1].reshape((img.shape[0],1))

    for left in range(size[1][0]):
        img = np.hstack((img_left, img))
    for right in range(size[1][1]):
        img = np.hstack((img, img_right))
    for up in range(size[0][0]):
        img = np.vstack((img_up, img))
    for down in range(size[0][1]):
        img = np.vstack((img, img_down))
    
    return img

def _padding_V(img, size):
    img_left = img[:,0].reshape((img.shape[0],1)).copy()
    img_right = img[:,-1].reshape((img.shape[0],1)).copy()

    for left in range(size):
        img = np.hstack((img_left, img))
    for right in range(size):
        img = np.hstack((img, img_right))

    return img

def _padding_H(img, size):
    img_up = img[0,:].copy()
    img_down = img[-1,:].copy()

    for up in range(size):
        img = np.vstack((img_up, img))
    for down in range(size):
        img = np.vstack((img, img_down))
    
    return img


def cross_correlation_1d(img, kernel):
    filter_size = kernel.shape[0]
    output = np.zeros(img.shape)
    vertical = False

    if kernel.ndim ==1:
        vertical = True
        padded_img = _padding_V(img, filter_size//2)
    else:
        kernel = np.squeeze(kernel)
        padded_img = _padding_H(img, filter_size//2)

    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):

            if vertical:
                width_range = (j, j + filter_size)
                this_conv = padded_img[i, width_range[0]:width_range[1]]
                out = (this_conv*kernel).sum()
            else:
                
                hight_range = (i, i + filter_size)
                this_conv = padded_img[hight_range[0]:hight_range[1], j]
                out = (this_conv*kernel).sum()
            output[i,j] = out

    return output

def cross_correlation_2d(img, kernel):
    filter_size = kernel.shape[0]
    output = np.zeros(img.shape)
    padded_img = _padding(img, ((filter_size//2, filter_size//2), (filter_size//2, filter_size//2)))

    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            width_range = (i, i + filter_size)
            hight_range = (j, j + filter_size)
            
            this_conv = padded_img[width_range[0]:width_range[1],hight_range[0]:hight_range[1]]
            out = (np.multiply(this_conv, kernel)).sum()
            output[i, j] = out
    return output

def _gaussian(x, y, sigma):
    inside_exp = -((x**2 + y**2)/(2*(sigma**2)))
    return 1/(2 * math.pi * sigma**2) * math.exp(inside_exp)

def get_gaussian_filter_1d(size, sigma):
    #size must be odd number
    output = np.ones(size)
    for i in range(size):
        in_g = i - size//2
        output[i] = _gaussian(in_g, 0, sigma)
    return output/output.sum()
    
def get_gaussian_filter_2d(size, sigma):
    #size must be odd number
    output = np.ones([size, size])
    for i in range(size):
        for j in range(size):
            x_i = i - size//2
            y_j = j - size//2
            output[i][j] = _gaussian(x_i, y_j, sigma)
    return output/output.sum()

def make_9images(img, file_name):
    #9개짜리 이미지 만들기
    output = None
    for sig in [1, 6, 11]:
        prev = None
        for siz in [5, 11, 17]:
            kernel_2d = get_gaussian_filter_2d(siz, sig)
            filtered_img = cross_correlation_2d(img, kernel_2d).astype(np.uint8)
            filtered_img = cv2.putText(filtered_img, "{}x{} s = {}".format(siz, siz, sig) , (5,40), 0,1,(0, 0, 0),2)
            if str(type(prev)) == "<class 'NoneType'>":
                prev = filtered_img
            else:
                prev = np.vstack((prev,filtered_img))

        if str(type(output)) == "<class 'NoneType'>":
            output = prev
        else:
            output = np.hstack((output,prev))
    show_and_save(output, f"part_1_gaussian_filtered_{file_name}")

if __name__ == "__main__":

    os.makedirs("./result", exist_ok= True)

    
    #가우시안 커널 구현후 프린트
    size = 5; sigma = 1
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    print("1D Gaussian Kernel")
    print(kernel_1d)
    kernel_2d = get_gaussian_filter_2d(size, sigma)
    print("2D Gaussian Kernel")
    print(kernel_2d)
    


    # Lenna에 대한 코드 
    #9개의 이미지를 보여주고 저장하는 코드
    img_lenna = load_lenna()
    make_9images(img_lenna, "lenna")

    # 사이즈17, 시그마6 짜리의 필터를 적용하여 difference map과 걸린 시간, absolute diffenece를 출력합니다.
    size = 17; sigma = 6
    print("\npart1_gaussian_filtered_lenna.png")
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    time_g1d = time.time()
    filtered_img_using1d = cross_correlation_1d(img_lenna, kernel_1d)
    filtered_img_using1d = cross_correlation_1d(filtered_img_using1d, kernel_1d.reshape((size, 1)))
    time_elap_1d = (time.time() - time_g1d)
    
    kernel_2d = get_gaussian_filter_2d(size, sigma)
    time_g2d = time.time()
    filtered_img_using2d = cross_correlation_2d(img_lenna, kernel_2d)
    time_elap_2d = (time.time() - time_g2d)
    print("{}x{} s={}\n1d를 사용하여 걸린시간 : {}\n2d를 사용하여 걸린시간 : {}".format(size, size, sigma, time_elap_1d, time_elap_2d))
    Difference_Map = filtered_img_using1d-filtered_img_using2d
    print("Absolute Difference of lenna : {}".format(np.absolute(Difference_Map).sum()))
    show(Difference_Map, "Difference Map of lenna")




    # shape에 대한 코드입니다.
    img_shape = load_shape()
    make_9images(img_shape, "shapes")

    size = 17; sigma = 6
    print("\npart_1_gaussian_filtered_shapes.png")
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    time_g1d = time.time()
    filtered_img_using1d = cross_correlation_1d(img_shape, kernel_1d)
    filtered_img_using1d = cross_correlation_1d(filtered_img_using1d, kernel_1d.reshape((size, 1)))
    time_elap_1d = (time.time() - time_g1d)

    kernel_2d = get_gaussian_filter_2d(size, sigma)
    time_g2d = time.time()
    filtered_img_using2d = cross_correlation_2d(img_shape, kernel_2d)
    time_elap_2d = (time.time() - time_g2d)
    print("{}x{} s={}\n1d를 사용하여 걸린시간 : {}\n2d를 사용하여 걸린시간 : {}".format(size, size, sigma, time_elap_1d, time_elap_2d))

    Difference_Map = filtered_img_using1d-filtered_img_using2d
    print("Absolute Difference of shapes : {}".format(np.absolute(Difference_Map).sum()))
    show(Difference_Map, "Difference Map of shapes")
