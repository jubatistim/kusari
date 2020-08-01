import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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
    blur_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

    # canny_img = cv2.Canny(blur_img, 50, 150)
    canny_img = cv2.Canny(blur_img, 50, 150)

    return canny_img

def region_focus(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height-30), (590, height-30), (330, 180)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def show_lines(image, lines):
    l_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(l_image, (x1, y1), (x2, y2), (255, 0, 0), 10) # THIS CHANGES VISIBILITY OF THE LINE
    return l_image

def build_coordinates(image, line_params):
    slope, yc = line_params
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - yc) / slope)
    x2 = int((y2 - yc) / slope)
    return np.array([x1, y1, x2, y2])

def slope_average(image, lines):
    l_fit = []
    r_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        params = np.polyfit((x1, x2), (y1, y2), 1)
        slope = params[0]
        yc = params[1]
        if slope < 0:
            l_fit.append((slope, yc))                
        else:
            r_fit.append((slope, yc))                
    
    l_fit_avg = np.average(l_fit, axis=0)
    r_fit_avg = np.average(r_fit, axis=0)

    left_line = build_coordinates(image, l_fit_avg)
    right_line = build_coordinates(image, r_fit_avg)

    return np.array([left_line, right_line])

### TEST SINGLE FRAME ###
# img = cv2.imread('VIDEO/vlcsnap-2020-07-25-23h08m06s816.png')

# img = imgResize(img)

# a = canny(img)
# b = region_focus(a)

# cv2.imshow('result', b)
# cv2.waitKey(0)
### TEST SINGLE FRAME ###

cap = cv2.VideoCapture('VIDEO/SEQ1 Copy 01.mp4')
just_line_success = None

while(cap.isOpened()):

    _, frame = cap.read()
    frame = imgResize(frame)

    if just_line_success is None:
        just_line_success = np.zeros_like(frame)

    try:     

        canny_img = canny(frame)

        masked_image = region_focus(canny_img)

        lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

        slope_avg_img = slope_average(frame, lines)

        just_lines = show_lines(frame, slope_avg_img)

        merged_image = cv2.addWeighted(frame, 0.8, just_lines, 1, 1)

        # plt.imshow(masked_image)
        # plt.show()

        cv2.imshow('result', merged_image)
        if cv2.waitKey(1) == ord('q'):
            break

        just_line_success = just_lines
        print('SUCCESS')

    except:

        merged_image_success = cv2.addWeighted(frame, 0.8, just_line_success, 1, 1)

        cv2.imshow('result', merged_image_success)
        if cv2.waitKey(1) == ord('q'):
            break

        print('ERROR')

cap.release()
cv2.destroyAllWindows()