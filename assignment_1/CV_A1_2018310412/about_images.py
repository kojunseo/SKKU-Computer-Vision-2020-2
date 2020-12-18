
import cv2
import time
import numpy as np
def show(img, name = "Temp Name"):
    cv2.imshow(f"{name}", (img).astype(np.uint8))
    cv2.waitKey()


def save(img, name):
    cv2.imwrite(f"./result/{name}.png",(img).astype(np.uint8))


def show_and_save(img, name):
    show(img, name)
    save(img, name)

def show_and_save_mag(mag, name):
    mag = np.where(mag<0, 0, mag)
    mag = np.where(mag>255, 255, mag)
    show(mag, name)
    save(mag, name)


def load_lenna():
    PATH_lenna = "./lenna.png"
    img_lenna = cv2.imread(PATH_lenna, cv2.IMREAD_GRAYSCALE)    
    return img_lenna


def load_shape():
    PATH_shape = "./shapes.png"
    img_shape = cv2.imread(PATH_shape, cv2.IMREAD_GRAYSCALE)
    return img_shape

