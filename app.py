import cv2
import numpy as np
import matplotlib.pyplot as plt

# IMAGE RESIZE METHOD
def imgResize(img, max_wh = 700):
    if img.shape[1] > img.shape[0]:
        if img.shape[1] > max_wh:
            n_width = max_wh
            n_height = max_wh * img.shape[0] / img.shape[1]

            img = cv2.resize(img, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
    else:
        if img.shape[0] > max_wh:
            n_height = max_wh
            n_width = max_wh * img.shape[1] / img.shape[0]

            img = cv2.resize(img, (int(n_width), int(n_height)), interpolation = cv2.INTER_AREA)
    return img

def canny(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # canny_img = cv2.Canny(blur_img, 50, 150)
    canny_img = cv2.Canny(blur_img, 50, 240)

    return canny_img

img = cv2.imread('VIDEO/vlcsnap-2020-07-25-23h08m06s816.png')

lanes_img = np.copy(img)

canny_img = canny(lanes_img)



plt.imshow(canny_img)
plt.show()