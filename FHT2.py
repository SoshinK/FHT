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

def FHT_vert_r(img):
    height = img.shape[0]   # высота картинки (ее делим пополам на каждой итерации)
    length = img.shape[1]   # длина картинки
    channels = img.shape[2]
    
    hough_Img = np.array([ [ [0] * channels for i in range(length)] for j in range(height)])
    if(height == 1):
        return img
    
    top = img[:(height // 2)]   # верх картинки
    bottom = img[(height // 2):] # низ картинки
    
    hough_top = FHT_vert_r(top)       # Хаф-образы верха и низа соответсвенно
    hough_bottom = FHT_vert_r(bottom) #
    
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

def FHT_vert_l(img):
    height = img.shape[0]   # высота картинки (ее делим пополам на каждой итерации)
    length = img.shape[1]   # длина картинки
    channels = img.shape[2]
    
    hough_Img = np.array([ [ [0] * channels for i in range(length)] for j in range(height)])
    if(height == 1):
        return img
    
    top = img[:(height // 2)]   # верх картинки
    bottom = img[(height // 2):] # низ картинки
    
    hough_top = FHT_vert_l(top)       # Хаф-образы верха и низа соответсвенно
    hough_bottom = FHT_vert_l(bottom) #
    
    for ch in range(channels):
        for i in range(height - 1, 0, -2):
            shift = height // 2 - 1 - (i - 1) // 2 # смещение
            for j in range(length):
                index = j - shift
                if(index >= length):
                    index = index % length
                
                hough_Img[i][j][ch] = int(hough_bottom[(i - 1) // 2][j][ch]) + int(hough_top[(i - 1) // 2][index][ch])
                index = j - shift - 1
                if(index >= length):
                    index = index % length
                    
                hough_Img[i - 1][j][ch] = int(hough_bottom[(i - 1) // 2][j][ch]) + int(hough_top[(i - 1) // 2][index][ch])
    return hough_Img

def FHT_horiz_l(img):
    height = img.shape[0]   # высота картинки (ее делим пополам на каждой итерации)
    length = img.shape[1]   # длина картинки
    channels = img.shape[2]
    
    hough_Img = np.array([ [ [0] * channels for i in range(length)] for j in range(height)])
    if(length == 1):
        return img
    
    left = []
    right = []
    for i in range(height):
        left.append(img[i][:(length // 2)])
        right.append(img[i][(length // 2):])
        #print(">>", img[i][:(length // 2)])
    hough_top = FHT_horiz_l(np.array(left))       # Хаф-образы верха и низа соответсвенно
    hough_bottom = FHT_horiz_l(np.array(right)) #
    
    for ch in range(channels):
        for i in range(length - 1, 0, -2):
            shift = length // 2 - 1 - (i - 1) // 2 # смещение
            for j in range(height):
                index = j + shift
                if(index >= height):
                    index = index % height
                
                hough_Img[j][i][ch] = int(hough_bottom[j][(i - 1) // 2][ch]) + int(hough_top[index][(i - 1) // 2][ch])
                index = j + shift + 1
                if(index >= height):
                    index = index % height
                    
                hough_Img[j][i - 1][ch] = int(hough_bottom[j][(i - 1) // 2][ch]) + int(hough_top[index][(i - 1) // 2][ch])
    return hough_Img

def FHT_horiz_r(img):
    height = img.shape[0]   # высота картинки (ее делим пополам на каждой итерации)
    length = img.shape[1]   # длина картинки
    channels = img.shape[2]
    
    hough_Img = np.array([ [ [0] * channels for i in range(length)] for j in range(height)])
    if(length == 1):
        return img
    
    left = []
    right = []
    for i in range(height):
        left.append(img[i][:(length // 2)])
        right.append(img[i][(length // 2):])
        #print(">>", img[i][:(length // 2)])
    hough_top = FHT_horiz_r(np.array(left))       # Хаф-образы верха и низа соответсвенно
    hough_bottom = FHT_horiz_r(np.array(right)) #
    
    for ch in range(channels):
        for i in range(length - 1, 0, -2):
            shift = length // 2 - 1 - (i - 1) // 2 # смещение
            for j in range(height):
                index = j - shift
                if(index >= height):
                    index = index % height
                
                hough_Img[j][i][ch] = int(hough_bottom[j][(i - 1) // 2][ch]) + int(hough_top[index][(i - 1) // 2][ch])
                index = j - shift - 1
                if(index >= height):
                    index = index % height
                    
                hough_Img[j][i - 1][ch] = int(hough_bottom[j][(i - 1) // 2][ch]) + int(hough_top[index][(i - 1) // 2][ch])
    return hough_Img


a = np.array([
              [[0], [0], [1], [0]],
              [[0], [0], [1], [0]],
              [[0], [1], [0], [0]],
              [[0], [1], [0], [0]]
              ])

e=np.array([
        [[1], [0]],
        [[1], [0]],
        [[0], [0]],
        [[0], [0]]
        ])    
    
img1 = cv.imread("./img/lisa64x64.png")
img2 = cv.imread("./img/line16x16purple.png")
img3 = cv.imread("./img/line64x64.png")
img4 = cv.imread("./img/point64x64.png")
img5 = cv.imread("./img/intersec_lines64x64purple.png")
img6 = cv.imread("./img/non-diadic_line64x64purple.png")
img7 = cv.imread("./img/line64x64_l.png")
img8 = cv.imread("./img/line64x64_horiz_r.png")
#plt.imshow(img6)
img9 = cv.imread("./img/qqq.png")
b=FHT_horiz_l(a)
print(e.shape)
for i in range(4):
    for j in range(4):
        print(b[i][j], end=' ')
    print('\n')
himg = FHT_vert_r(img3)
himg = normalize2(himg)
plt.imshow(himg)
