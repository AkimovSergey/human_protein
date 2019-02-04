import cv2
import numpy as np
from PIL import Image
from os import path, mkdir, rename, listdir
import imutils
import pandas as pd
import scipy
from utils import *

data_path = 'E:\\all\\'
train_path = path.join(data_path, 'train')

def find_centers_from_all_colors(img_path):
    image = np.array(Image.open(img_path + '_blue.png'))
    image += np.array(Image.open(img_path + '_green.png'))
    image += np.array(Image.open(img_path + '_red.png'))
    image += np.array(Image.open(img_path + '_yellow.png'))
    find_centers(image)
    '''cv2.imshow('test', image)
    cv2.waitKey(0)'''


def get_possible_mats(img_path, max=10):
    res = []
    image_blue_ch = np.array(Image.open(img_path + '_blue.png'))
    centers = find_centers(image_blue_ch)
    dst = find_centers_distance(centers)
    for center in centers:
        max -= 1
        if max <= 0:
            break
    return res


def get_mat(img, pos, size):
    x = int(pos[1])
    y = int(pos[0])
    dst = int(size/2)
    res = img[x - dst:x + dst, y - dst:y + dst]

    ''' try:
        res = cv2.resize(res, (int(out_size/2), out_size))
    except Exception as e:
        print(e)'''
    #cv2.imshow("Image", res)
    #cv2.waitKey(0)
    return res

def get_mats(img, pos, size, size_res):
    res = []
    for y in range(pos[0] - int(size/2), pos[0] + int(size/2), size_res):
        for x in range(pos[1] - int(size / 2), pos[1] + int(size / 2), size_res):
            res.append(img[x:x + size_res, y:y + size_res])
    return res

def get_3_mat(img, pos, size, size_res):
    res = []
    x = int(pos[1])
    y = int(pos[0])
    dst = int(size/2)
    dst2 = int(size_res/2)
    if size_res >= size:
        res.append(img[x - dst2:x + dst2, y - dst2:y + dst2])
        res.append(img[x - dst2:x + dst2, y - dst2:y + dst2])
        res.append(img[x - dst2:x + dst2, y - dst2:y + dst2])
        return res
    # left-bottom corner
    res.append(img[x - dst:x - dst + size_res, y + dst - size_res:y + dst])
    # center
    res.append(img[x - dst2:x + dst2, y - dst2:y + dst2])
    # top-right corner
    res.append(img[x + dst - size_res:x + dst, y - dst:y - dst + size_res])

    ''' try:
        res = cv2.resize(res, (int(out_size/2), out_size))
    except Exception as e:
        print(e)'''
    #cv2.imshow("Image", res)
    #cv2.waitKey(0)
    return res


def find_centers_distance(centers):
    dst = scipy.spatial.distance.cdist(centers, centers)
    dst = [np.min(x[x > 30]) for x in dst]
    return np.mean(dst)


def get_random_position(img):
    centers = find_centers(img=img)
    dst = find_centers_distance(centers)
    sz = img.shape[0]
    while True:
        center = centers[np.random.randint(0, len(centers))]
        if dst < center[1] < sz - dst and center[0] < sz - dst:
            break
        dst -= 1
    return center, dst


def find_centers(img):
    bright = 50
    sz = 25
    msz = SZ/2
    res = []
    cntss = []
    while sz > 5:
        blurred = cv2.GaussianBlur(img, (sz, sz), 0)
        thresh = cv2.threshold(blurred, bright, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if bright == 20:
            sz -= 4
            bright = 50
        else:
            bright -= 10

        if len(cnts) != 0:
            cntss.append(cnts)

    if len(cnts) == 0:
        return res
    cnts = max(cntss, key=len)

    imsh = img.copy()
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0.0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        if msz < cX < (IM_SZ - msz) and msz < cY < (IM_SZ - msz):
            res.append((cX, cY))

        '''cv2.drawContours(imsh, [c], -1, (0, 255, 0), 2)
        cv2.circle(imsh, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(imsh, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('tt', imsh)
    cv2.waitKey(0)'''

    return res


def merge_images_from_numbers():
    for idx in range(28):
        dirt = path.join(train_path, str(idx))
        if path.exists(dirt):
            for file in listdir(dirt):
                rename(path.join(dirt, file), path.join(train_path, file))


def split_images_by_number():
    data = pd.read_csv(path.join(data_path, 'train.csv'))
    for name, labels in zip(data['Id'], data['Target'].str.split(' ')):
        idxs = np.array([int(label) for label in labels])
        if len(idxs) == 1:
            dirt = path.join(train_path, str(idxs[0]))
            if not path.exists(dirt):
                mkdir(dirt)
            for append in ['green', 'blue', 'red', 'yellow']:
                nm = name + '_' + append + '.png'
                rename(path.join(train_path, nm), path.join(dirt, nm))


def centers_to_square(file_name):
    file = path.join(train_path, file_name #'c4482b38-bbbc-11e8-b2ba-ac1f6b6435d0'
                    #'004d8a0e-bbc4-11e8-b2bc-ac1f6b6435d0'
                     )
    image_blue_ch = np.array(Image.open(file + '_blue.png'))
    image_red_ch = np.array(Image.open(file + '_red.png'))
    dtr = 64
    centers = find_centers(image_blue_ch, dtr)

    for center in centers:
        if dtr < center[1] > 512 - dtr or dtr < center[0] > 512 - dtr:
            continue
        img = image_red_ch[center[1] - dtr: center[1] + dtr, center[0] - dtr: center[0] + dtr]
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        dst = cv2.logPolar(img, (dtr, dtr), dtr/3, cv2.INTER_LINEAR)
        cv2.imshow("Image", dst)
        cv2.waitKey(0)


def gather_statistic_cells_size():
    for data_id in data['Id']:
        image_path = path.join(train_path, data_id)
        image_blue_ch = np.array(Image.open(image_path + '_blue.png'))
        sz = image_blue_ch.shape[0]
        centers = find_centers(image_blue_ch)
        if len(centers) == 0:
            print('centers not found. id = {0}'.format(data_id))
            continue
        dst = find_centers_distance(centers)
        if dst > 150:
            print('big distance. id = {0}'.format(data_id))




#gather_statistic_cells_size()
'''file = path.join(train_path, '000a9596-bbc4-11e8-b2bc-ac1f6b6435d0')
find_centers_from_all_colors(file)'''

file = path.join(train_path, 'ac809320-bbbf-11e8-b2bb-ac1f6b6435d0')
image_blue_ch = np.array(Image.open(file + '_blue.png'))
find_centers(image_blue_ch)


# centers_to_square('004d8a0e-bbc4-11e8-b2bc-ac1f6b6435d0')
#merge_images_from_numbers()

'''img = convolve_2d_mean(image_blue_ch)
img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(np.uint8)

cv2.imshow('wnd', img)
cv2.waitKey(0)'''

