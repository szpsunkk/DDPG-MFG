# import cv2 as cv

import os

import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class map():
    def __init__(self, path):
        self.path = path
        # self.title = title

    def plot(self):
        img = plt.imread(self.path)
        # img = plt.imread('D:\\MEC\\MEC\\map\\123.png')
        fig = plt.figure('show picture')

        # cmap指定为'gray'用来显示灰度图, 1 1 1
        ax = fig.add_subplot(111)
        ax.imshow(img)
        # 以灰度图显示图片, 对彩色图像没有作用
        # ax.imshow(img, cmap='gray')
        ax.set_title("Vehicle edge computing (VEC) for digital twin")  # 给图片加titile

        # 直接显示一张图片,是使用下面这种方法,等效于之前操作ax的4条语句
        # plt.imshow(img)
        plt.axis('off')  # 不显示刻度

        plt.show()
