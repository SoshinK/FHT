#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 17:22:42 2018

Бытсрое преобразование Хафа для преимущественно-вертикальных
прямых с наклоном вверх

@author: soshink
"""

import numpy as np

def FHT(img):
    height = img.shape[0]   # высота картинки (ее делим пополам на каждой итерации)
                            
    length = img.shape[1]   # длина картинки
    
    hough_Img = np.array([[0] * length for i in range(height)]) 
    
    if(height == 1):
        return img
    
    top = img[:(height // 2)]   # верх картинки
    bottom = img[(height // 2):] # низ картинки
    
    hough_top = FHT(top)       # Хаф-образы верха и низа соответсвенно
    hough_bottom = FHT(bottom) #


    """
    Идем шагом по -2 с конца массива (то есть снизу вверх) итогового Хаф-образа
    На i-ую строчку записываем сумму соответсвующей (т.е. (i - 1) / 2 -ой) строчки
    из Хаф-образа нижней половины и (i - 1) / 2 -ой строчки из Хаф-образа
    верхней половины, сдвинутой на некоторое смещение shift, равное наклону нижнего
    соответствующего паттерна.
    На (i-1)-ую строчку записываем сумму соответсвующей (т.е. (i - 1) / 2 -ой) строчки
    из Хаф-образа нижней половины и (i - 1) / 2 -ой строчки из Хаф-образа
    верхней половины, сдвинутой на некоторое смещение shift + 1.
    """
    for i in range(height - 1, 0, -2):
        shift = height // 2 - 1 - (i - 1) // 2 # смещение
        for j in range(length):
            index = j + shift
            if(index >= length):
                index = index % length
            
            hough_Img[i][j] = hough_bottom[(i - 1) // 2][j] + hough_top[(i - 1) // 2][index]
            
            index = j + shift + 1
            if(index >= length):
                index = index % length
                
            hough_Img[i - 1][j] = hough_bottom[(i - 1) // 2][j] + hough_top[(i - 1) // 2][index]
            
    
    return hough_Img
        




a = np.array([
              [0, 1, 0, 1],
              [1, 0, 1, 2],
              [1, 1, 1, 1],
              [2, 2, 2, 2]
              ])
b = np.array([
              [0, 1, 0, 1, 1, 1, 0, 1],
              [1, 0, 1, 2, 0, 1, 0, 1],
              [1, 1, 1, 1, 0, 1, 2, 1],
              [2, 2, 2, 2, 0, 1, 0, 1],
              [0, 1, 0, 1, 1, 1, 2, 1],
              [1, 0, 1, 2, 5, 1, 0, 1],
              [1, 1, 1, 1, 0, 1, 0, 1],
              [2, 2, 2, 2, 3, 1, 0, 1]
              ])
c = np.random.randint(0, 20, (16, 16))

print(FHT(a))
print(FHT(b))
print(FHT(c))
