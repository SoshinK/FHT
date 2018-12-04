#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:31:16 2018

@author: soshink
"""
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def normalize1(img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for c in range(img.shape[2]):
                img[i][j][c] = np.uint8(img[i][j][c] % 256)
    return img

def normalize2(img):
    for c in range(img.shape[2]):
        max = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if max < img[i][j][c]:
                    max = img[i][j][c]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i][j][c] = np.uint8(round(img[i][j][c] / max * 255))
    return img

def FHT(img):
    height = img.shape[0]   # высота картинки (ее делим пополам на каждой итерации)
    length = img.shape[1]   # длина картинки
    channels = img.shape[2]
    
    hough_Img = np.array([ [ [0] * channels for i in range(length)] for j in range(height)])
    if(height == 1):
        return img
    
    top = img[:(height // 2)]   # верх картинки
    bottom = img[(height // 2):] # низ картинки
    
    hough_top = FHT(top)       # Хаф-образы верха и низа соответсвенно
    hough_bottom = FHT(bottom) #
    
    for ch in range(channels):
        for i in range(height - 1, 0, -2):
            shift = height // 2 - 1 - (i - 1) // 2 # смещение
            for j in range(length):
                index = j + shift
                if(index >= length):
                    index = index % length
                
                hough_Img[i][j][ch] = int(hough_bottom[(i - 1) // 2][j][ch]) + int(hough_top[(i - 1) // 2][index][ch])
                index = j + shift + 1
                if(index >= length):
                    index = index % length
                    
                hough_Img[i - 1][j][ch] = int(hough_bottom[(i - 1) // 2][j][ch]) + int(hough_top[(i - 1) // 2][index][ch])
    return hough_Img


img1 = cv.imread("./lisa64x64.png")
img2 = cv.imread("./line16x16purple.png")
#plt.imshow(img1)
himg = FHT(img2)
himg = normalize2(himg)
plt.imshow(himg)
