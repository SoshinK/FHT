#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 02:29:08 2019

@author: kvsoshin
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import copy
from math import log2
from math import floor
from math import ceil



"""
Нормализация изображения. В каждом канале ищется самое большое значение пикселя
и делается равным 255. Остальные значения делятся на это наибольшее и умножаются на 
255.

Аргументы:
    img - исходное изображение numpy.array
Возвращаемое значение:
    Нормализованное изображение numpy.array
"""
def hough_normalize(img):
    height = img.shape[0]
    length = img.shape[1]
    channels = img.shape[2]
    
    res = np.array([ [ [np.uint8(0)] * channels for i in range(length)] for j in range(height)])
    for c in range(img.shape[2]):
        max = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if max < img[i][j][c]:
                    max = img[i][j][c]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                res[i][j][c] = np.uint8(round(img[i][j][c] / max * 255))
    return res




"""
Ищет точку на изображении с наибольшей суммой значении по всем каналам.

Аргументы:
    himg - исходное изображение numpy.array
Возвращаемое значение:
    пара координат пикселя (int,int)
"""
def hough_getmaxpnt(himg):
    height = himg.shape[0]
    length = himg.shape[1]
    channels = himg.shape[2]
    max = 0
    a = 0
    maxx = 0
    maxy = 0
    for i in range(height):
        for j in range(length):
            a = 0
            for c in range(channels):
                a += himg[i][j][c]
            if (a > max):
                max = a
                maxx = i
                maxy= j
    return maxx, maxy




"""
Преобразование Хафа квадратного изображения для вертикальных-правых прямых.
Т.е. угол между осью Ox и прямой составляет от pi/4 до pi/2.

Аргументы:
    img - исходное изображение numpy.array
Возвращаемое значение:
    Хаф-образ numpy.array
"""
def FHT_vert_r(img):
    height = img.shape[0]
    length = img.shape[1]
    channels = img.shape[2]
    hough_Img = np.array([ [ [0] * channels for i in range(length)] for j in range(height)])
    if(length == 1):
        return img
    left = []
    right = []
    for i in range(height):
        left.append(img[i][:(length // 2)])
        right.append(img[i][(length // 2):])
    left = FHT_vert_r(np.array(left))
    right = FHT_vert_r(np.array(right))
    for ch in range(channels):
        for i in range(0, length, 2):
            shift = i // 2
            for j in range(height):
                index = (j + shift) % height
                hough_Img[j][i][ch] = int(left[j][i // 2][ch]) + int(right[index][i // 2][ch])
                index = (j + shift + 1) % height
                hough_Img[j][i + 1][ch] = int(left[j][i // 2][ch]) + int(right[index][i // 2][ch])
    return hough_Img




"""
Отражение изображения относительно оси, параллельной Oy.

Аргументы:
    img - исходное изображение numpy.array
Возвращаемое значение:
    Отраженное изображение numpy.array
"""
def reflect_y(img):
    img = copy.deepcopy(img)
    for i in range(img.shape[0] // 2):
        tmp = copy.deepcopy(img[i])
        img[i] = img[img.shape[0] - i - 1]
        img[img.shape[0] - i - 1] = tmp
    return img





"""
Отражение изображения относительно оси, параллельной Ox

Аргументы:
    img - исходное изображение numpy.array
Возвращаемое значение:
    Отраженное изображение numpy.array
"""
def reflect_x(img):
    img = copy.deepcopy(img)
    for i in range(img.shape[0]):
        img[i] = reflect_y(img[i])
    return img




"""
Преобразование Хафа квадратного изображения для указанных прямых.

Аргументы:
    img - исходное изображение numpy.array
    direction - принимает возможные значения:
        vr - преобразование Хафа для вертикальных-правых прямых.
            Т.е. угол между осью Ox и прямой составляет от pi/4 до pi/2.
        vl - преобразование Хафа для вертикальных-левых прямых.
            Т.е. угол между осью Ox и прямой составляет от -pi/2 до -pi/4.
        hd - преобразование Хафа для горизонтальных-нижних прямых.
            Т.е. угол между осью Ox и прямой составляет от -pi/4 до 0.
        hu - преобразование Хафа для горизонтальных-верхних прямых.
            Т.е. угол между осью Ox и прямой составляет от 0 до pi/4.
        all - преобразование Хафа для всех наклонов прямых.
Возвращаемое значение:
    Хаф-образ numpy.array. Если direction='all', то Хаф-образ будет состоять из
    четырех сложенных подряд Хаф-образов(слева направо): vl, vr, hu, hd, то есть
    размер образа будет n*4n, где n - длина стороны исходного изображения. При
    остальных значениях direction на выходе будут соответствующие Хаф-образа
    размером n*n.
"""
def FHT(img, direction='all'):
    height = img.shape[0]
    length = img.shape[1]
    channels = img.shape[2]
    
    if(direction == 'all' or direction == 'vr'):
        hough_vr = FHT_vert_r(img)
        ret = hough_vr
        
    if(direction == 'all' or direction == 'vl'):
        hough_vl = reflect_x(reflect_y(FHT_vert_r(reflect_y(img))))
        ret = hough_vl
        
    if(direction == 'all' or direction == 'hd'):
        hough_hd = np.transpose(reflect_y(FHT_vert_r(reflect_y(np.transpose(img ,(1, 0, 2))))), (1, 0, 2))
        
        #поворот на 90 против ч
        hough_hd = np.transpose(hough_hd, (1, 0, 2))
        hough_hd = reflect_y(hough_hd)
        
        ret = hough_hd
        
    if(direction == 'all' or direction == 'hu'):
        hough_hu = np.transpose(FHT_vert_r(np.transpose(img, (1, 0, 2))), (1, 0, 2))
        hough_hu = reflect_y(hough_hu) 
        #поворот на 90 против ч
        hough_hu = np.transpose(hough_hu, (1, 0, 2))
        hough_hu = reflect_y(hough_hu)
        ret = hough_hu
        
    if(direction == 'all'):
        hough_all = np.array([ [ [0] * channels for i in range(4 * length)] for j in range(height)])      
        for i in range(height):
            hough_all[i][:length] = (hough_vl[i])
            hough_all[i][length:(2 * length)] = hough_vr[i]
            hough_all[i][(2 * length):(3 * length)] = hough_hu[i]
            hough_all[i][(3 * length):(4 * length)] = hough_hd[i]
  
        ret = hough_all
    return hough_normalize(ret)




"""
Математическое округление в большую сторону.

Аргументы:
    n - действительное
Возвращаемое значение:
    Округленное значение.
"""
def hough_rd(n):
    if((n - int(n)) >= 0.5):
        return (int(n)+1)
    else:
        return int(n)





"""
Нарисовать вертикальный-правый диадический паттерн.
Аргументы:
    img - исходное изображение numpy.array
    s - величина наклона паттерна. Модуль разницы абсцисс верхней и нижней точек.
    t - смещение начала. Абсцисса нижней точки.
    color - цвет прямой. Одномерный numpy.array
Возвращаемое значение:
    Изображение с прямой numpy.array
"""
def draw_vert_r(img, s, t, color):
    img = copy.deepcopy(img)
    p = int(log2(img.shape[0]))
    t_bytes = np.zeros(p, dtype=int)
    for i in range(p):
        t_bytes[i] = t % 2
        t = t // 2
    for y in range(img.shape[0]):
        x = 0
        for k in range(len(t_bytes)):
            x += t_bytes[k] * hough_rd(2 ** k * y/ (img.shape[0] - 1))
        x += s
        x = x % img.shape[0]
        img[x][y] = np.uint8(color)
    return img




"""
Нарисовать набор прямых(диадических паттернов) разных наклонов данным цветом 
на данном изображении img.

Аргументы:
    img - исходное изображение numpy.array
    points - массив двумерных массивов соответствующий прямым (значения
        s и t - наклон и смещение (см. draw_vert_r)), которые надо нарисовать. 
        Причем, t может принимать значения от 0 до n, где n - размер img. 
        s принимает значения от 0 до 4*n. Причем, если если t < n, то 
        соответствующие s и t  кодируют вертикальную-левую прямую, если n < t < 2n, 
        то вертикально-правую, если 2n < t < 3n, то горизонтальную-верхнюю, 
        если 3n < t < 4n, то горизонтальную-нижнюю - то есть расположение, идентичное
        Хаф-образу, возвращаемому FHT с опцией directions='all'. points соответствует
        тому, что возвращает функция hough_lines.
    color - цвет прямой. Одномерный numpy.array с длиной равной числу каналов.
Возвращаемое значение:
    Изображение с прямой numpy.array
"""
def draw_diadlns(img, points, color):
    length = img.shape[1]
    height = img.shape[0]
    canvas = copy.deepcopy(img)
    for i in range(len(points)):
        if(points[i][1] in range(length)):
            canvas = reflect_y(canvas)
            canvas = draw_vert_r(canvas, height - points[i][0] - 1, height - points[i][1], color)
            canvas = reflect_y(canvas)
            
        if(points[i][1] in range(length, 2 * length)):
            canvas = draw_vert_r(canvas, points[i][0], points[i][1], color)
            
        if(points[i][1] in range(length * 2, 3 * length)):
            canvas = np.transpose(canvas, (1, 0, 2))
            canvas = reflect_y(canvas)
            canvas = reflect_y(canvas)
            canvas = draw_vert_r(canvas, height - points[i][0] - 1, height - points[i][1] - 1, color)
            canvas = reflect_y(canvas)
            canvas = reflect_y(canvas)
            canvas = np.transpose(canvas, (1, 0, 2))
            
        if(points[i][1] in range(length * 3, 4 * length)):
            canvas = np.transpose(canvas, (1, 0, 2))
            canvas = reflect_y(canvas)
            canvas = draw_vert_r(canvas, points[i][0], points[i][1], color)
            canvas = reflect_y(canvas)
            canvas = np.transpose(canvas, (1, 0, 2))
    return canvas    





"""
Нарисовать на данном изображении img квадрат центр которого лежит в точке (x, y).
Функция исользуется что бы подавить уже найденные яркие точки на Хаф-образе.

Аргументы:
    img - исходное изображение numpy.array
    x - абсцисса центра квадрата
    y - ордината центра квадрата
    sq_size - расстояние от центра квадрата до стороны(по умолчанию 4).
    color - цвет прямоугольника. Одномерный numpy.array с длиной равной числу каналов
Возвращаемое значение:
    Изображение numpy.array с прямоугольником
"""
def supp_point(img, x, y, sq_size = 6, color = (0, 0, 0)):
    img = copy.deepcopy(img)
    for i in range (x - sq_size, x + sq_size + 1, 1):
        for j in range (y - sq_size, y + sq_size + 1, 1):
            if(i >= 0 and i < img.shape[0] and j >= 0 and j < img.shape[1]):
                img[i][j] = color
    return img






"""
Найти на изображении несколько прямых любого наклона.

Аргументы:
    img - исходное изображение numpy.array
    numlns - число прямых
    sq_size - см. supp_point. Исользуется для отметки уже найденных прямых по
        Хаф-образу. По умолчанию 4.
Возвращаемое значение:
    Массив прямых в формате points - см. draw_diadlns.
"""        
def hough_getlines(img, numlns, sq_size = 6):
    retlines = []
    canvas = hough_normalize(cv.Laplacian(img, cv.CV_64F))
    himg = FHT(canvas)
    for i in range(numlns):
        line = hough_getmaxpnt(himg)
        himg = supp_point(himg, line[0], line[1], sq_size, (0, 0, 0))
        retlines.append(line)
    return retlines




def byterepr(val, digits):
    k = val
    brepr = np.zeros(digits, dtype=int)
    for i in range(digits):
        brepr[digits - i - 1] = k % 2
        k = k // 2
    if k == 0:
        return ''.join(list(map(str, brepr)))



def get_diadln(imgsize, point1, point2):
    if(point2[1] >= point1[1]):
        low = point1
        high = point2
    else:
        low = point2
        high = point1
    p = int(log2(imgsize))
    delta_x = high[0] - low[0]
    highb = byterepr(high[1], p)
    lowb = byterepr(low[1], p)
    prev_sum = 0
    diadpatternsp_h = []
    diadpatternsp_l = []
    for i in range(p):
        diadpatternsp_h.append(int(highb[:(i + 1)], 2) - prev_sum)
        prev_sum += diadpatternsp_h[i]
    prev_sum = 0
    for i in range(p):
        diadpatternsp_l.append(int(lowb[:(i + 1)], 2) - prev_sum)
        prev_sum += diadpatternsp_l[i]
    diad_diff = np.array(diadpatternsp_h) - np.array(diadpatternsp_l)
    bytes_t = np.zeros(p, dtype=int)
    for i in range(p - 1, -1, -1):
        if (diad_diff[i] <= delta_x):
            bytes_t[i] = 1
            delta_x -= diad_diff[i]
    
    t = 0
    bpow = 1
    for i in range(p):
        t += bytes_t[i] * bpow
        bpow *= 2
    shift = 0
    prev_sum = 0
    for i in range(p):
        a = int(lowb[:(i + 1)], 2)
        a -= prev_sum
        prev_sum += a
        shift += a * bytes_t[i]
    s = low[0] - shift
    return s, t

    



        
            