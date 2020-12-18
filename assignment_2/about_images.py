import cv2
import numpy as np
def load_smile():
    img = cv2.imread("./smile.png", cv2.IMREAD_GRAYSCALE)
    return img

def load_pair_book():
    img1 = cv2.imread("./cv_cover.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./cv_desk.png", cv2.IMREAD_GRAYSCALE)
    return img2,img1 

def load_poter():
    img1 = cv2.imread("./hp_cover.jpg", cv2.IMREAD_GRAYSCALE)
    # return img1
    return cv2.resize(img1, dsize=(350,440), interpolation=cv2.INTER_AREA) 


def load_pair_diamondhead():
    img1 = cv2.imread("./diamondhead-10.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("./diamondhead-11.png", cv2.IMREAD_GRAYSCALE)
    return img1, img2


def show(img, name = "Temp Name"):
    cv2.imshow(f"{name}", (img).astype(np.uint8))
    cv2.waitKey()


def save(img, name):
    cv2.imwrite(f"./result/{name}.png",(img).astype(np.uint8))


def show_and_save(img, name):
    show(img, name)
    save(img, name)

if __name__=="__main__":
    print(load_pair_book()[0].shape)
    print(load_pair_book()[1].shape)