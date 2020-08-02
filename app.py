import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# HYPERPARAMETERS
GAUSSIAN_BLUR_KERNEL_SIZE = 9 # 5
GAUSSIAN_BLUR_STD = 0 # 0

CANNY_TRESHOLD_1 = 50 # 50
CANNY_TRESHOLD_2 = 150 # 150

FOCUS_EDGE1_X = 0.29
FOCUS_EDGE1_Y = 0.92
FOCUS_EDGE2_X = 0.84
FOCUS_EDGE2_Y = 0.92
FOCUS_EDGE3_X = 0.47
FOCUS_EDGE3_Y = 0.47

LINE_COLOR = (255, 0, 0)
LINE_THICKNESS = 10

LIMIT_SCREEN_FOR_LINES = 3/5

RIGHT_SLOPE_LIMIT = 0.09
RIGHT_SLOPE_STANDARD = 0.63
RIGHT_STANDARD_Y = 0

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

    blur_img = cv2.GaussianBlur(gray_img, (GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_KERNEL_SIZE), GAUSSIAN_BLUR_STD)

    canny_img = cv2.Canny(blur_img, CANNY_TRESHOLD_1, CANNY_TRESHOLD_2)

    return canny_img

def region_focus(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(int(FOCUS_EDGE1_X*width), int(FOCUS_EDGE1_Y*height)), (int(FOCUS_EDGE2_X*width), int(FOCUS_EDGE2_Y*height)), (int(FOCUS_EDGE3_X*width), int(FOCUS_EDGE3_Y*height))]
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
            cv2.line(l_image, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS) # THIS CHANGES VISIBILITY OF THE LINE
    return l_image

def build_coordinates(image, line_params):
    slope, yc = line_params
    y1 = image.shape[0]
    y2 = int(y1*(LIMIT_SCREEN_FOR_LINES))
    x1 = int((y1 - yc) / slope)
    x2 = int((y2 - yc) / slope)
    return np.array([x1, y1, x2, y2])

previous_left_line = np.array([0,0,0,0])
previous_right_line = np.array([0,0,0,0])
def slope_average(image, lines):
    global previous_left_line
    global previous_right_line
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
            if slope < RIGHT_SLOPE_LIMIT:
                slope = RIGHT_SLOPE_STANDARD
                yc = RIGHT_STANDARD_Y
            r_fit.append((slope, yc))          
    
    l_fit_avg = np.average(l_fit, axis=0)
    r_fit_avg = np.average(r_fit, axis=0)

    if np.isnan(l_fit_avg).any():
        left_line = previous_left_line
    else:
        left_line = build_coordinates(image, l_fit_avg)
        previous_left_line = left_line

    if np.isnan(r_fit_avg).any():
        right_line = previous_right_line
    else:
        right_line = build_coordinates(image, r_fit_avg)
        previous_right_line = right_line

    return np.array([left_line, right_line])

### TEST SINGLE FRAME ###
# img = cv2.imread('VIDEO/vlcsnap-2020-08-01-17h01m17s903.png')
# img = imgResize(img)

# canny_img = canny(img)

# masked_image = region_focus(canny_img)

# lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# slope_avg_img = slope_average(img, lines)

# just_lines = show_lines(img, slope_avg_img)

# merged_image = cv2.addWeighted(img, 0.8, just_lines, 1, 1)

# cv2.imshow('result', merged_image)
# cv2.waitKey(0)
### TEST SINGLE FRAME ###

cap = cv2.VideoCapture('VIDEO/SEQ1 Copy 01.mp4')
just_line_success = None

while(cap.isOpened()):

    _, frame = cap.read()
    frame = imgResize(frame, 1000)

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