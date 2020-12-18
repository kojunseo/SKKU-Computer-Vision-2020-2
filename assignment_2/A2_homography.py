from about_images import *
import cv2
import numpy as np
import random
import time

def BFMatcher_match(desc1, desc2):
    objList = []
    squares = np.array([2**i for i in range(9)]).reshape([9,1])
    for idx1, d1 in enumerate(desc1):
        obj = cv2.DMatch()
        distList = [ [(lambda x, y: np.count_nonzero((np.unpackbits(x.reshape(x.shape[0], 1), axis = 1) == np.unpackbits(y.reshape(y.shape[0], 1), axis = 1)) == False))(d1, d2), idx1, idx2] for idx2, d2 in enumerate(desc2)  ] #해밍거리 구하기 (람다처리)
        obj.distance, obj.queryIdx, obj.trainIdx = min(distList, key = lambda x:x[0])
        objList.append(obj)
    return sorted(objList, key = lambda x: x.distance)

def MATCHER_RATIO(desc1, desc2, th = 0.8):
    objList = []
    squares = np.array([2**i for i in range(9)]).reshape([9,1])
    for idx1, d1 in enumerate(desc1):
        obj = cv2.DMatch()
        distList = [ [(lambda x, y: np.count_nonzero((np.unpackbits(x.reshape(x.shape[0], 1), axis = 1) == np.unpackbits(y.reshape(y.shape[0], 1), axis = 1)) == False))(d1, d2), idx1, idx2] for idx2, d2 in enumerate(desc2)  ]
        sorted_list = sorted(distList, key = lambda x:x[0])
        obj.distance, obj.queryIdx, obj.trainIdx = sorted_list[0]
        ratio = sorted_list[0][0] / sorted_list[1][0]
        objList.append([obj, ratio])
    
    after_sort = np.array(sorted(objList, key= lambda x: x[0].distance))
    
    final = []
    for i in after_sort:
        if i[1] < th: # 제일 큰거랑 그 다음거 비율이 th 이하인 것들만 처리해줌
            final.append(i[0])
    return final


def Norm_matrix(matrix):    
    mean_x, mean_y = np.mean(matrix, axis = 0)
    sub_M = np.array([[1,0,-mean_x],[0,1,-mean_y],[0,0,1]])
    sub_matrix = matrix - np.array([mean_x, mean_y])

    hy = np.hypot(sub_matrix[:,0], sub_matrix[:,1])
    # print(np.sqrt(2), np.max(hy))
    scaling = np.sqrt(2)/np.max(hy)
    scale_M = np.array([[scaling,0,0],[0,scaling,0],[0,0,1]])

    return np.dot(scale_M, sub_M)


def Get_P(points1, points2, Lists):
    scr_P = np.array([[points1[Lists[i].queryIdx].pt[0],points1[Lists[i].queryIdx].pt[1]] for i in range(len(Lists))])
    dest_P = np.array([[points2[Lists[i].trainIdx].pt[0],points2[Lists[i].trainIdx].pt[1]] for i in range(len(Lists))])
    return np.array(scr_P), np.array(dest_P)


def Normalized_P(points1, points2, Lists, N = 15):
    scr_P = np.array([[points1[Lists[i].queryIdx].pt[0],points1[Lists[i].queryIdx].pt[1]] for i in range(N)])
    dest_P = np.array([[points2[Lists[i].trainIdx].pt[0],points2[Lists[i].trainIdx].pt[1]] for i in range(N)])
    T_s, T_d = Norm_matrix(scr_P), Norm_matrix(dest_P)
    Normed_scr_P = [np.dot(T_s, vector)[:2]/np.dot(T_s, vector)[2] for vector in np.hstack((scr_P, np.ones([scr_P.shape[0], 1])))]
    Normed_dest_P = [np.dot(T_d, vector)[:2]/np.dot(T_d, vector)[2] for vector in np.hstack((dest_P, np.ones([dest_P.shape[0], 1])))]

    return np.array(Normed_scr_P), np.array(Normed_dest_P), T_s, T_d

def compute_homography(src_P, dest_P):
    concated_A = []

    for i in range(len(src_P)):
        x = src_P[i,0]
        y = src_P[i,1]
        x_p = dest_P[i,0]
        y_p = dest_P[i,1]
        concated_A.append([-x, -y, -1, 0, 0, 0, x*x_p, y*x_p, x_p])
        concated_A.append([0,0,0,-x, -y, -1, x*y_p, y*y_p, y_p])
        
    concated_A = np.array(concated_A)
    _, s, vh = np.linalg.svd(concated_A)
    h = vh[-1].reshape([3,3])
    
    return h /h[2,2]

def compute_homography_ransac(srcP, destP, th):
    np.seterr(all='ignore')
    start = time.time()
    whole = np.array([i for i in range(len(srcP))])

    inlier = -1
    while time.time() - start < 3:
        random_four = np.random.choice(whole, size = 4, replace=False)
        
        H_curr = compute_homography(np.array([srcP[i] for i in random_four]), np.array([destP[i] for i in random_four]))
        reproj = np.array([np.dot(H_curr, vector)[:2]/np.dot(H_curr, vector)[2] for vector in np.hstack((srcP, np.ones((srcP.shape[0], 1))))])
        dist_ = np.hypot((destP - reproj)[:,0],(destP - reproj)[:,1])
        potential_inlier = np.sum(dist_ < th)
        if potential_inlier > inlier:
            inlier = potential_inlier
            H = H_curr
            potential_inlier_src = srcP[dist_<th]
            potential_inlier_dst = destP[dist_<th]
    return H
        

def wraping(under, cover):
    output = np.where(cover == 0, under, cover)
    return output

def stitching(H, img1, img2):
    RU = (img1.shape[1], 0, 1)
    point = np.array(np.dot(H, RU)[:2]/np.dot(H, RU)[2])
    cuted_img2 = img2[:, int(round(point[0])):]

    show(np.hstack([img1, cuted_img2]), "Image Stitching")

def stitching_gradation(H, img1, img2, gradation_range = 100):
    RU = (img1.shape[1], 0, 1)

    point = np.array(np.dot(H, RU)[:2]/np.dot(H, RU)[2])
    cuted_img2 = img2[:, int(round(point[0]))-gradation_range:]
    blend_left = np.array([i/gradation_range for i in range(gradation_range, 0, -1)])
    blend_right = np.array([i/gradation_range for i in range(0, gradation_range)])
    for line in range(len(img1)):
        img1[line, -gradation_range:] = img1[line, -gradation_range:]* blend_left + cuted_img2[line, :gradation_range] * blend_right

    cuted_img2 = img2[:, int(round(point[0])):]

    show(np.hstack([img1, cuted_img2]), "Image Stitching with Gradation")
    
    

if __name__ == "__main__":
    img1, img2 = load_pair_book()
    orb = cv2.ORB_create()
    kp1 = orb.detect(img1, None)
    kp1, des1 = orb.compute(img1, kp1)

    kp2 = orb.detect(img2, None)
    kp2, des2 = orb.compute(img2, kp2)

    # 2 - 1 
    matchingList = BFMatcher_match(des1, des2)
    show(cv2.drawMatches(img1,kp1,img2,kp2,matchingList[:10],None,flags=2),"Draw Matches")

    # 2 - 2
    matchingList = MATCHER_RATIO(des1, des2, th = 0.8)
    srcP, destP, T_s, T_d = Normalized_P(kp1, kp2, matchingList, N = 15)
    H_n = compute_homography(destP, srcP)
    H = np.dot(np.dot(np.linalg.inv(T_s),H_n),T_d)
    des_change = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    show(wraping(img1, des_change), "Normalized Homography, warping")


    matchingList = MATCHER_RATIO(des1, des2, th = 0.8)
    srcP, destP= Get_P(kp1, kp2, matchingList[:14])
    H_n = compute_homography_ransac(destP, srcP, th = 4)
    des_change = cv2.warpPerspective(img2, H_n, (img1.shape[1], img1.shape[0]))
    show(wraping(img1, des_change), "RANSAC Homography, warping")


    img_p = load_poter()
    des_change = cv2.warpPerspective(img_p, H_n, (img1.shape[1], img1.shape[0]))
    show(wraping(img1, des_change), "RANSAC with porter book")


    # 2- 5 Image stitching
    img_ten, img_elv = load_pair_diamondhead()
    kp_ten = orb.detect(img_ten, None)
    kp_ten, des_ten = orb.compute(img_ten, kp_ten)

    kp_elv = orb.detect(img_elv, None)
    kp_elv, des_elv = orb.compute(img_elv, kp_elv)

    matchingList = MATCHER_RATIO(des_ten, des_elv, th=0.8)
    srcP_ten, destP_elv = Get_P(kp_ten, kp_elv, matchingList[:13])
    
    H_n = compute_homography_ransac(srcP_ten, destP_elv , th = 3)

    stitching(H_n, img_ten, img_elv )
    stitching_gradation(H_n, img_ten, img_elv, gradation_range = 100 )



