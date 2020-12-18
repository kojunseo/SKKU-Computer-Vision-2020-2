#2018310412 인공지능융합전공 고준서
from about_images import *
from compute_avg_reproj_error import compute_avg_reproj_error
import numpy as np
import cv2
import time
np.seterr(divide='ignore', invalid='ignore')

def compute_F_raw(M):
    concated_A = np.zeros((len(M), 9))
    
    for idx, (x,y,x_p,y_p) in enumerate(M):
        concated_A[idx] = [x*x_p, x*y_p, x, y*x_p, y*y_p, y, x_p, y_p, 1]
    _, s, vh = np.linalg.svd(concated_A)
    F = vh[-1].reshape([3,3])
    return F

def ComputeNormMatrix(M):
    global img_height, img_width
    img_h, img_w = img_height, img_width
    img_h /= 2 
    img_w /= 2
    transform = np.array([[1, 0, -img_w],
                         [0, 1, -img_h],
                         [0, 0, 1]])
    scaling = np.array([[1/img_w, 0, 0],
                       [0, 1/img_h, 0],
                       [0, 0, 1]])
    return np.dot(scaling, transform)

def ChangeNormMatrix(M, _matrix):
    left, right = M[:, :2], M[:,2:]
    return np.hstack([[np.dot(_matrix, vector)[:2]/np.dot(_matrix, vector)[2] for vector in np.hstack((left, np.ones([left.shape[0], 1])))], [np.dot(_matrix, vector)[:2]/np.dot(_matrix, vector)[2] for vector in np.hstack((right, np.ones([right.shape[0], 1])))]])

def compute_F_norm(M):
    matrix = ComputeNormMatrix(M)
    M_ = ChangeNormMatrix(M, matrix)
    
    F_ = compute_F_raw(M_)
    U,S,Vt = np.linalg.svd(F_)
    S[-1] = 0
    F_ = np.dot(U, np.dot(np.diag(S), Vt))
    return np.dot(np.dot(matrix.T, F_), matrix)

def compute_F_mine(M, get_ = 8):
    start = time.time()
    matrix = ComputeNormMatrix(M)
    M_ = ChangeNormMatrix(M, matrix)
    min_error = 10
    min_F_ = None
    while True:
        choice = np.random.choice(np.arange(0, len(M_)), get_, replace=False)
        picked = M_[choice]
        concated_A = np.zeros((len(M_), 9))
        for idx, (x,y,x_p,y_p) in enumerate(picked):
            concated_A[idx] = [x*x_p, x*y_p, x, y*x_p, y*y_p, y, x_p, y_p, 1]
        _, s, vh = np.linalg.svd(concated_A)
        F_ = vh[-1].reshape([3,3])

        U,S,Vt = np.linalg.svd(F_)
        S[-1] = 0
        F_ = np.dot(U, np.dot(np.diag(S), Vt))
        
        care = compute_avg_reproj_error(M_, F_)
        if min_error > care:
            min_error = care
            min_F_ = F_
        
        if  time.time() - start >= 2.9:
            break

    F_ = min_F_
    return np.dot(np.dot(matrix.T, F_), matrix)

def pointForLine(F, pt1s, endpoint):
    re = []
    for p1, p2 in pt1s:
        re.append(np.dot(F, np.array([p1, p2, 1]).reshape((3,1))))
    return np.array(re)

def drawLines(img1, img2, lines, pts1, pts2):
    r, c, _ = img1.shape
    color = ((255,0,0),(0,255,0), (0,0,255))
    image1, image2 = img1.copy(), img2.copy()
    for idx, (r, pt1, pt2) in enumerate(zip(lines, pts1, pts2)):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c,-(r[2]+r[0]*c)/r[1]])
        cv2.line(image1, (x0, y0), (x1, y1), color[idx], 2)
        cv2.circle(image1, tuple(pt1), 5, color[idx], -1)
        cv2.circle(image2, tuple(pt2), 5, color[idx], -1)
    
    return image1, image2


def VisualizeEpipolarLine(M, F1, F2, img1, img2):
    while True:
        a1, a2, a3 = np.random.choice(np.arange(0, len(M)), 3, replace=False)
        pt1s = np.array([M[a1][:2].astype("int"), M[a2][:2].astype("int"), M[a3][:2].astype("int")])
        pt2s = np.array([M[a1][2:].astype("int"), M[a2][2:].astype("int"), M[a3][2:].astype("int")])

        lin1 = pointForLine(F1, pt1s, img1.shape[0])
        lin2 = pointForLine(F2, pt2s, img2.shape[0])

        image1 , _ = drawLines(img1, img2, lin2.reshape(-1, 3), pt1s, pt2s)
        image2, _ = drawLines(img2, img1, lin1.reshape(-1, 3), pt2s, pt1s)
        cv2.imshow("Epipolar Lines",  np.hstack((image1, image2)).astype(np.uint8))
        a = cv2.waitKey()
        if a == ord("q"):
            cv2.destroyAllWindows()
            return

def req_error(name, M, get_ = 10):
    print(f"Average Reprojection Errors ({name})")
    F = compute_F_raw(M)
    print("Raw =", compute_avg_reproj_error(M, F))
    
    F = compute_F_norm(M)
    print("Norm =", compute_avg_reproj_error(M, F))

    F = compute_F_mine(M, get_)
    # F2 = compute_F_mine(np.hstack((M[:,2:], M[:,:2])), get_)
    print("Mine =", compute_avg_reproj_error(M, F))


if __name__ =="__main__":
    # 1-1 compute F with 3 methods
    M_temple = np.loadtxt( "temple_matches.txt" )
    temple1, temple2 = pair_templates()
    img_height, img_width, _ = temple1.shape
    req_error("temple1.png, temple2.png", M_temple, get_=8)
    

    M_lib = np.loadtxt( "library_matches.txt" )
    lib1, lib2 = pair_library()
    img_height, img_width, _ = lib1.shape
    req_error("library1.png, library2.png", M_lib, get_ = 8)

    M_house = np.loadtxt( "house_matches.txt" )
    house1, house2 = pair_house()
    img_height, img_width, _ = house1.shape
    req_error("house1.png, house2.png", M_house, get_ = 8)


    #1-2 draw epipolar lines
    img_height, img_width, _ = temple1.shape
    F1 = compute_F_mine(M_temple, 8)
    F2 = compute_F_mine(np.hstack([M_temple[:, 2:],M_temple[:, :2]]), 8)
    VisualizeEpipolarLine(M_temple, F1, F2, temple1, temple2)

    img_height, img_width, _ = lib1.shape
    F1 = compute_F_mine(M_lib, 8)
    F2 = compute_F_mine(np.hstack([M_lib[:, 2:],M_lib[:, :2]]), 8)
    VisualizeEpipolarLine(M_lib, F1, F2, lib1, lib2)

    img_height, img_width, _ = house1.shape
    F1 = compute_F_mine(M_house, 8)
    F2 = compute_F_mine(np.hstack([M_house[:, 2:],M_house[:, :2]]), 8)
    VisualizeEpipolarLine(M_house, F1, F2, house1, house2)