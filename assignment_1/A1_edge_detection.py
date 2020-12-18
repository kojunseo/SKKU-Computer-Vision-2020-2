from A1_image_filtering import cross_correlation_2d, get_gaussian_filter_2d
import cv2
from about_images import show_and_save, load_lenna, load_shape, show_and_save_mag
from computational_time import TimeModule
import numpy as np
import math

def non_maximun_suppression_dir(mag, dir):
    dir = np.rad2deg(dir) + 180
    window_size = 3
    term = window_size//2
    output = mag.copy()

    for i in range(term, mag.shape[0]-term):
        for j in range(term, mag.shape[1]-term):
            angle = dir[i,j]
            
            if ((0 <= angle <= 22.5) or (337.5 < angle <= 360)) or (157.5 < angle <= 202.5) : 
                if mag[i,j] <= mag[i,j+1] or mag[i,j] < mag[i,j-1]:
                    output[i,j] = 0

            elif (22.5 < angle <= 67.5) or (202.5< angle <= 247.5): 
                if mag[i,j] < mag[i-1,j-1] or mag[i,j] <= mag[i+1,j+1]:
                    output[i,j] = 0

            elif (67.5 < angle <= 112.5) or (247.5< angle <= 292.5): 
                if mag[i,j] <=  mag[i+1,j] or mag[i,j] < mag[i-1,j]:
                    output[i,j] = 0
            
            elif (112.5< angle <= 157.5) or (292.5< angle <= 337.5): 
                if mag[i,j] <= mag[i-1,j+1] or mag[i,j] < mag[i+1,j-1]:
                    output[i,j] = 0
    output = np.where(output<0, 0, output)
    output = np.where(output>255, 255, output)
    return output


def compute_image_gradient(img):
    S_x = np.array([[-1,0,1],
                    [-2,0,2], 
                    [-1,0,1]])
    S_y = S_x.T
    sobel_x = cross_correlation_2d(img, S_x)
    sobel_y = cross_correlation_2d(img, S_y)

    direction = np.arctan2(sobel_y, sobel_x)
    magnitude = np.hypot(sobel_x, sobel_y)

    return magnitude, direction



if __name__ == "__main__":

    img_shape = load_shape()
    img_lenna = load_lenna()
     
    gaussian_filter = get_gaussian_filter_2d(7, 1.5)
    filtered_shape = cross_correlation_2d(img_shape, gaussian_filter)
    filtered_lenna = cross_correlation_2d(img_lenna, gaussian_filter)


    t = TimeModule()
    mag_lenna, dir_lenna = compute_image_gradient(filtered_lenna)
    t.end_print("image gradient lenna")
    show_and_save_mag(mag_lenna, "part_2_edge_raw_lenna")

    t = TimeModule()
    mag_shapes, dir_shapes = compute_image_gradient(filtered_shape)
    t.end_print("image gradient shapes")
    show_and_save_mag(mag_shapes, "part_2_edge_raw_shapes")
    

    t = TimeModule()
    nms_lenna = non_maximun_suppression_dir(mag_lenna, dir_lenna)
    t.end_print("nms shapes")
    show_and_save(nms_lenna, "part_2_edge_sup_lenna")


    t = TimeModule()
    nms_shape = non_maximun_suppression_dir(mag_shapes, dir_shapes) 
    t.end_print("ms shapes")
    show_and_save(nms_shape, "part_2_edge_sup_shapes")
