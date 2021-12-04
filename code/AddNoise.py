import cv2
import numpy as np


def gauss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise * 1
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    #cv.imshow("gasuss", out)
    return out

img = cv2.imread("img.BMP", cv2.IMREAD_GRAYSCALE)
cv2.imwrite("img.png", img)

# cv2.imshow("img", img)
# cv2.waitKey(0)

noise_img = gauss_noise(img, 0, 0.001)
# cv2.imwrite("noise_img.png", noise_img)

cv2.imshow("Nimg", noise_img)
cv2.waitKey(0)