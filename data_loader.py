import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from utils import show_image
from investigate import *

data_path = 'E:\\all\\'
expls_path = os.path.join(data_path, 'expls')

class DataGenerator:

    is_encoder_train = True

    @staticmethod
    def get_image_arrays(path, center, size=SZ):
        try:
            image_blue_ch = get_mats(np.array(Image.open(path + '_blue.png')), center, size, KERNEL_SZ)
            image_red_ch = get_mats(np.array(Image.open(path + '_red.png')), center, size, KERNEL_SZ)
            image_green_ch = get_mats(np.array(Image.open(path + '_green.png')), center, size, KERNEL_SZ)
            image_yellow_ch = get_mats(np.array(Image.open(path + '_yellow.png')), center, size, KERNEL_SZ)
            image = np.concatenate((image_blue_ch, image_red_ch, image_green_ch, image_yellow_ch), axis=0)
            image = np.swapaxes(image, 0, 2)
        except:
            return None

        return image

    @staticmethod
    def get_image_3_array(path, center, size):
        try:
            image_blue_ch = get_3_mat(np.array(Image.open(path + '_blue.png')), center, size, KERNEL_SZ)
            image_red_ch = get_3_mat(np.array(Image.open(path + '_red.png')), center, size, KERNEL_SZ)
            image_green_ch = get_3_mat(np.array(Image.open(path + '_green.png')), center, size, KERNEL_SZ)
            image_yellow_ch = get_3_mat(np.array(Image.open(path + '_yellow.png')), center, size, KERNEL_SZ)
            image = []

            '''image_green_c_ = get_mat(np.array(Image.open(path + '_green.png')), center, size)
            cv2.imshow('im', image_green_c_)
            cv2.imshow('im_orig', np.array(Image.open(path + '_green.png')))'''
            '''cv2.imshow('im1', image_green_ch[0])
            cv2.imshow('im2', image_green_ch[1])
            cv2.imshow('im3', image_green_ch[2])
            cv2.waitKey(0)'''

            '''for i in range(len(image_blue_ch)):
                image.append(np.stack((
                    image_green_ch[i],
                    image_blue_ch[i],
                    image_red_ch[i],
                    image_yellow_ch[i]), -1))'''

            image = np.stack((image_green_ch[0],
                                image_blue_ch[0],
                                image_red_ch[0],
                                image_yellow_ch[0],
                                image_green_ch[1],
                                image_blue_ch[1],
                                image_red_ch[1],
                                image_yellow_ch[1],
                                image_green_ch[2],
                                image_blue_ch[2],
                                image_red_ch[2],
                                image_yellow_ch[2]), -1)

        except:
            return None
        return np.array(image)


    @staticmethod
    def get_image_array(path, center=None, size=None):
        try:
            if not size:
                size = SZ
            if not center:
                y = x = np.random.randint(size/2, IMSZ - size/2)
                center = (x, y)
            image_blue_ch = get_mat(np.array(Image.open(path + '_blue.png')), center, size)
            image_red_ch = get_mat(np.array(Image.open(path + '_red.png')), center, size)
            image_green_ch = get_mat(np.array(Image.open(path + '_green.png')), center, size)
            image_yellow_ch = get_mat(np.array(Image.open(path + '_yellow.png')), center, size)

            '''image = np.stack((
                image_red_ch,
                image_green_ch,
                image_blue_ch), -1)
            cv2.imwrite(os.path.join(expls_path, os.path.basename(path)) + '.png', image)'''

            '''cv2.imshow('ik', np.subtract(image_green_ch, image_blue_ch))
            cv2.waitKey(0)'''

            '''tp = np.subtract(image_green_ch, image_blue_ch)
            tp1 = np.subtract(image_green_ch, image_yellow_ch)
            tp2 = np.subtract(image_green_ch, image_red_ch)
            tp3 = np.subtract(image_blue_ch, image_yellow_ch)
            tp4 = np.subtract(image_blue_ch, image_red_ch)
            tp5 = '''

            '''image = np.stack((
                np.subtract(image_green_ch, image_blue_ch),
                np.subtract(image_green_ch, image_yellow_ch),
                np.subtract(image_green_ch, image_red_ch),
                np.subtract(image_blue_ch, image_yellow_ch),
                np.subtract(image_blue_ch, image_red_ch),
                np.subtract(image_yellow_ch, image_green_ch),
                np.subtract(image_yellow_ch, image_blue_ch),
                np.subtract(image_yellow_ch, image_red_ch)), -1)'''

            image = np.stack((
                            image_green_ch,
                            image_yellow_ch,
                            image_red_ch,
                            image_blue_ch), -1)
        except Exception as e:
            print(e)
            return None
        return image

    @staticmethod
    def load_image(image_path, use_dst=False):
        image_blue_ch = np.array(Image.open(image_path + '_blue.png'))
        center = None
        size = SZ
        try:
            center, size = get_random_position(image_blue_ch)
        except: pass
        if center is None:
            return None
        if use_dst:
            return DataGenerator.get_image_arrays(image_path, center, size)
        else:
            return DataGenerator.get_image_arrays(image_path, center)


    @staticmethod
    def create_train(dataset, batch_size):
        while True:
            batch_images = np.empty((batch_size, KERNEL_SZ, KERNEL_SZ, 64))
            batch_labels = np.zeros((batch_size, 28))


            for i in range(batch_size):
                rand_type = dataset[np.random.randint(0, len(dataset))]
                image = None
                item = None
                while image is None:
                    item = rand_type[np.random.randint(0, len(rand_type))]
                    image = DataGenerator.load_image(item['path'])

                batch_images[i] = image
                batch_labels[i][item['labels']] = 1
            if DataGenerator.is_encoder_train:
                yield batch_images, batch_images
            else:
                yield batch_images, batch_labels

    @staticmethod
    def load_set_from_image(image_path):
        res = []
        image_blue_ch = np.array(Image.open(image_path + '_blue.png'))
        sz = image_blue_ch.shape[0]
        centers = find_centers(image_blue_ch)
        if len(centers) < 2:
            image_yellow_ch = np.array(Image.open(image_path + '_yellow.png'))
            centers = find_centers(image_yellow_ch)
            if len(centers) < 2:
                centers = [[256, 256]]
        dst = find_centers_distance(centers)
        for center in centers:
            if dst < center[1] < sz - dst and dst < center[0] < sz - dst:
                res.append(DataGenerator.get_image_arrays(image_path, center))
        return np.array(res)
