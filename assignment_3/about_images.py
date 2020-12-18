import cv2

def pair_templates():
    return cv2.imread("./temple1.png"), cv2.imread("./temple2.png")

def pair_library():
    return cv2.imread("./library1.jpg"), cv2.imread("./library2.jpg")

def pair_house():
    return cv2.imread("./house1.jpg"), cv2.imread("./house2.jpg")
