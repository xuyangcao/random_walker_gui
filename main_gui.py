# -*- coding: utf-8 -*-
import cv2 
import numpy as np 
from random_walker import random_walker
import pydicom
try:
    import pyamg
    amg_loaded = True
    print('pyamg loaded !')
except ImportError:
    amg_loaded = False
import time

from scipy.misc import bytescale
import matplotlib.pyplot as plt 

def normalize(img):
    '''
    加载数据后需要将图像归一化到(mean=0, std=1),否则可能导致程序出错.
    '''
    mean = np.mean(img)
    std = np.std(img)

    img = (img - mean) / std

    return img


def mouse_handler(event, x, y, flags, data) :
    '''
    响应鼠标的回调函数,用于画点,画完后按'Esc'结束.
    '''
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['img'], (x,y), 2, (0, 0, 255), 2)
        cv2.imshow("Image", data['img'])
        data['label'].append([x,y])
    
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON: 
        cv2.circle(data['img'], (x,y), 2, (0, 0, 255), 2)
        cv2.imshow("Image", data['img'])
        data['label'].append([x,y])

def get_labels(img, label):
    '''
    获取前景和背景点

    Input:
    ---
    position: str类型,可选`forground`或者`baclground`

    Output:
    ---
    一组标记点
    '''
    img = bytescale(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    data = {}
    data['img'] = img.copy()
    data['label'] = []

    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)

    print(data['label'])
    # Convert array to np.array
    points = np.vstack(data['label']).astype(np.uint8)

    return points

def main():
    # 1. 读入图像
    filename = './1.3.46.670589.11.38240.5.0.5472.2017092719265836760.dcm'
    ds = pydicom.dcmread(filename)
    img = ds.pixel_array

    # 2. 标记前景点和背景点,标记完成后按`Esc`结束标记.
    forground_labels = get_labels(img, 1)
    background_labels = get_labels(img, 0)

    # 3. 将手工获取的标记点整理为marker
    marker = np.zeros(img.shape, np.uint8)
    for label in forground_labels:
        marker[label[1], label[0]] = 1
    for label in background_labels:
        marker[label[1], label[0]] = 2
    # plt.figure(), plt.imshow(marker, 'gray')
    # plt.show()

    # 4. 数据预处理
    img = img.astype(np.float32)
    img = normalize(img)
    marker = marker.astype(np.float32)
    # print('img.max', img.max())

    # 5. Segmentation
    if amg_loaded:
        t1 = time.time()
        labels = random_walker(img, marker, mode='cg_mg', beta=50, tol=5.e-3)
        t2 = time.time()
        print (t2 - t1)
    else:
        labels = random_walker(img, marker, mode='cg', beta=100) 
    
    # 6. Show results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(img, cmap=plt.cm.gray, vmin=-2, vmax=2, interpolation='nearest')
    plt.contour(labels, [1.5])
    plt.title('img')
    plt.subplot(122)
    plt.imshow(marker, cmap=plt.cm.gray)
    plt.title('markers')
    plt.show()

if __name__ == '__main__':
    main()