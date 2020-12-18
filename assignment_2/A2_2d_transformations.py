import cv2
import numpy as np
from about_images import show, load_smile

def load_plane():
    plane = np.zeros([801,801]) + 255
    return plane

def process_arrow(img):
    cv2.arrowedLine(img, (0,400), (801,400), (0,0,0), 2, tipLength=0.02)
    cv2.arrowedLine(img, (400,801), (400,0) , (0,0,0), 2, tipLength=0.02)
    return img


def loadM(key):
    sin5 = -0.08715574274765817
    cos5 = 0.9961946980917455

    if key == ord("a"):
        return np.array([[1,0,-5],[0,1,0],[0,0,1]])

    elif key == ord("d"):
        return np.array([[1,0,5],[0,1,0],[0,0,1]])

    elif key == ord("w"):
        return np.array([[1,0,0],[0,1,-5],[0,0,1]])

    elif key == ord("s"):
        return np.array([[1,0,0],[0,1,5],[0,0,1]])

    
    elif key == ord("r"):
        return np.array([[cos5,-sin5,0],[sin5,cos5,0],[0,0,1]])
    
    elif key == ord("R"):
        return np.array([[cos5,sin5,0],[-sin5,cos5,0],[0,0,1]])


    elif key == ord("f"):
        return np.array([[-1,0,0],[0,1,0],[0,0,1]])
    
    elif key == ord("F"):
        return np.array([[1,0,0],[0,-1,0],[0,0,1]])
    
    elif key == ord("x"):
        return np.array([[0.95,0,0],[0,1,0],[0,0,1]])
    
    elif key == ord("X"):
        return np.array([[1.05,0,0],[0,1,0],[0,0,1]])
    
    elif key == ord("y"):
        return np.array([[1,0,0],[0,0.95,0],[0,0,1]])
    
    elif key == ord("Y"):
        return np.array([[1,0,0],[0,1.05,0],[0,0,1]])
    
    else:
        return np.array([[1,0,0],[0,1,0],[0,0,1]])


def get_transformed_image(img, M):
    plane = load_plane()
    width, hight = img.shape
    t_width = width//2
    t_hight = hight//2

    for i in range(-t_width, t_width+1):
        for j in range(-t_hight, t_hight+1):
            vector = [j, i, 1]
            doted = np.dot(M, vector)

            re_j, re_i , _ = doted/doted[2] + 400
            re_i = int(re_i)
            re_j = int(re_j)

            plane[re_i,re_j] = img[i+t_width,j+t_hight]

    
    return process_arrow(plane)



if __name__=="__main__":
    img = load_smile()
    M = loadM(00)
    plane = get_transformed_image(img, M)

    while True:
        cv2.imshow("2D Transformation", (plane).astype(np.uint8))
        key = cv2.waitKey()
        print(key)        
        if key == ord("Q"):
            cv2.destroyAllWindows()
            break
        elif key == ord("H"):
            M = loadM("i")
        else:
            M = np.dot(loadM(key), M)
        
        plane = get_transformed_image(img, M)
