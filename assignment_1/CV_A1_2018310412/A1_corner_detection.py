from A1_image_filtering import get_gaussian_filter_2d, cross_correlation_2d
from about_images import show_and_save, load_lenna, load_shape
from computational_time import TimeModule
import numpy as np
import cv2

def response_function(lamb_1, lamb_2, k = 0.04):
    return lamb_1*lamb_2 - k*((lamb_2*lamb_2)**2)


def sobel_filtering(img):
    S_x = np.array([[-1,0,1],
                    [-2,0,2], 
                    [-1,0,1]])
    S_y = S_x.T
    sobel_x = cross_correlation_2d(img, S_x)
    sobel_y = cross_correlation_2d(img, S_y)

    return sobel_x, sobel_y 

def compute_corner_response(img):
    window_size = 5
    return_img = np.zeros(img.shape)
    term = window_size//2
    Ix, Iy = sobel_filtering(img)
    IxIx = Ix **2
    IxIy = Ix * Iy
    IyIy = Iy **2

    for i in range(term, return_img.shape[0]-term):
        for j in range(term, return_img.shape[1]-term):
            CovIxIx = (IxIx[i-term:i+term+1, j-term:j+term+1]).sum()
            CovIxIy = (IxIy[i-term:i+term+1, j-term:j+term+1]).sum()
            CovIyIy = (IyIy[i-term:i+term+1, j-term:j+term+1]).sum()

            detM = CovIxIx*CovIyIy - np.square(CovIxIy)
            traceM = CovIxIx + CovIyIy
            return_img[i, j] = detM - 0.04*(pow(traceM, 2))
            

    return_img[return_img < 0] = 0
    Zmax , Zmin = np.amax(return_img), np.amin(return_img)
    Z = (return_img - Zmin)  / (Zmax - Zmin)
    return Z


def thresholding(org_img, corner_img):
    org_img = cv2.cvtColor(org_img,cv2.COLOR_GRAY2RGB)
    org_img[:,:,0] = np.where(corner_img >= 0.1, 0, org_img[:,:,0])
    org_img[:,:,1] = np.where(corner_img >= 0.1, 255, org_img[:,:,1])
    org_img[:,:,2] = np.where(corner_img >= 0.1, 0, org_img[:,:,2])
    return org_img


def thresholding_circle(org_img, corner_img):
    org_img = cv2.cvtColor(org_img,cv2.COLOR_GRAY2RGB)
    for i in range(org_img.shape[0]):
        for j in range(org_img.shape[1]):
            if corner_img[i, j] != 0:
                org_img = cv2.circle(org_img, (j, i), 5, (0,255,0), 2)
    return org_img


def non_maximun_supression_win(R, winSize):
    term = winSize //2
    R_c = R.copy()
    count = 0
    for i in range(term, R.shape[0]-term + 1):
        for j in range(term, R.shape[1]-term + 1):
            now_window = R[i-term:i+term+1, j-term:j+term+1]
            value = now_window[term, term]
            
            if value < now_window.max():
                R_c[i, j] = 0
                
    R_c = np.where(R_c <= 0.1 , 0, R_c)
    return R_c



if __name__ == "__main__":
    # 3-1 import images and Apply the Gaussian filtering to the input image.

    #load lenna - 직접 모듈을 만들어서 불러왔습니다.
    img_lenna = load_lenna()
    img_shape = load_shape()

    gaussian_filter = get_gaussian_filter_2d(7,1.5)
    filtered_lenna = cross_correlation_2d(img_lenna, gaussian_filter)
    filtered_shape = cross_correlation_2d(img_shape, gaussian_filter)

    # lenna에 대한 코드
    t = TimeModule()
    corner_lenna = compute_corner_response(filtered_lenna)
    t.end_print("compute_corner_response lenna")
    show_and_save(corner_lenna* 255, "part_3_corner_raw_lenna")   
   
    lenna_corner_bin = thresholding(img_lenna, corner_lenna)
    show_and_save(lenna_corner_bin, "part_3_corner_bin_lenna")

    t = TimeModule()
    corner_lenna = non_maximun_supression_win(corner_lenna, 11)
    t.end_print("nms of lenna")
    lenna_corner_sup = thresholding_circle(img_lenna, corner_lenna)
    show_and_save(lenna_corner_sup, "part_3_corner_sup_lenna")


    # shape에 대한 코드
    t = TimeModule()
    corner_shape = compute_corner_response(filtered_shape)
    t.end_print("compute_corner_response shapes")
    show_and_save(corner_shape * 255, "part_3_corner_raw_shapes")

    shape_corner_bin = thresholding(img_shape, corner_shape)
    show_and_save(shape_corner_bin, "part_3_corner_bin_shapes")
    
    t = TimeModule()
    corner_shape = non_maximun_supression_win(corner_shape, 11)
    t.end_print("nms of shapes")
    shape_corner_sup = thresholding_circle(img_shape, corner_shape)
    show_and_save(shape_corner_sup, "part_3_corner_sup_shape")


