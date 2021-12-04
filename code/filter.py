import cv2
import numpy as np
from copy import deepcopy

def mean_filter(img, window_sz):
    return cv2.blur(img, (window_sz, window_sz))



def Gauss_filter(img, window_sz):
    return cv2.GaussianBlur(img, (window_sz, window_sz), 0)

def threshold_mean_filter(img, window_sz=3, threshold=0):
    (height, width) = img.shape
    img_copy = deepcopy(img)
    for m in range(height):
        for n in range(width):
            try:
                sum = 0
                bias = int((window_sz - 1) / 2)
                for i in range(window_sz):
                    for j in range(window_sz):
                        sum += int(img[m-bias+i, n-bias+j])
                mean = sum / (window_sz * window_sz)
                if np.abs(img[m,n]-mean) > threshold:
                    img_copy[m, n] = mean
            except Exception as ex:
                # print(ex)
                pass
    return np.uint8(img_copy)

def half_mean_filter(img, threshold=0):
    (height, width) = img.shape
    img_copy = deepcopy(img)

    keys = [(1,2,3),(2,3,6),(3,6,9),(6,9,8),(9,8,7),(8,7,4),(7,4,1),(4,1,2)]
    cnt = 0
    for m in range(height):
        for n in range(width):
            try:
                values = []
                diffs = []
                M6 = []
                for i in range(3):
                    for j in range(3):
                        values.append(int(img_copy[m-1+i, n-1+j]))
                        # sum += int(img_copy[m-1+i, n-1+j])
                for k in range(8):
                    key = keys[k]
                    part1 = values[key[0]-1] + values[key[1]-1] + values[key[2]-1]
                    part2 = sum(values) - part1
                    diff = np.abs(part1 / 3 - part2 / 6)
                    diffs.append(diff)
                    M6.append(part2 / 6)
                max_diff = max(diffs)
                # print(max_diff)
                if max_diff > threshold:
                    cnt +=1
                    img_copy[m,n] = M6[diffs.index(max_diff)]
                else:
                    img_copy[m,n] = sum(values) / 9
            except Exception as ex:
                # print(ex)
                pass
    print("cnt=", cnt)
    return np.uint8(img_copy)




if __name__ == "__main__":
    img = cv2.imread("noise_img.png", cv2.IMREAD_GRAYSCALE)
    # result = threshold_mean_filter(img, 7, 15)
    # result = half_mean_filter(img, 5)
    # result = Gauss_filter(img, 9)
    result = half_mean_filter(img,20)
    cv2.imwrite("half_mean_thr20.png", result)
    cv2.imshow("result", result)
    cv2.waitKey(0)