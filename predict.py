import numpy as np
from yolo import YOLO
from PIL import Image, ImageDraw
import os
import math
# import cv2
import torch
import sys
from osgeo import gdal
from osgeo import osr

from utils.utils import bbox_IOU, Dataset, geo2lonlat

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
Image.MAX_IMAGE_PIXELS = None   #read big img
yolo = YOLO()

#-----load classification model-----------------#
model = torch.load('resnet18_sar_model.pth')
model.cuda()
model.eval()

# 以下代码演示读取tif的第一个通道的数据，并且获取经纬度信息
root_dir = os.getcwd()
path = os.path.join(root_dir, r'input/')
im_names = os.listdir(path)
im_names.sort()
for im_name in im_names:
    # dir_path = r"/home/pc001/e/gcgnew/yolov4-pytorch-master/GF3/"
    # filename = "GF3_SAY_SL_012236_E104.1_N1_msk_Cnv.tif"
    # file_path = os.path.join(dir_path, filename)
    src_path = os.path.join(path, im_name)
    dataset = Dataset(src_path)
    band = 1
    data, pielwidthxx, pixelheighty = dataset.get_data_and_resolution(band)  # 获取第一个通道的数据
    img = Image.fromarray(data)
    map_x, map_y = dataset.get_lon_lat()  # 获取经纬度信息

    dataset1 = gdal.Open(src_path)
    ct = geo2lonlat(dataset1)

    #-----------------------------------CROP-------------------------#
    save_sub_path = os.path.join(root_dir, r'output/crop_result/')#子图像保存路径
    # save_cls_path = os.path.join(root_dir, r'output/cls_result/')#最终分类结果
    save_det_path =os.path.join(root_dir, r'/output/')#最终分类结果
    w, h= img.size #获得图像尺寸
    resolution = min(abs(pielwidthxx), abs(pixelheighty))
    overlap = 300//resolution #重叠的像素个数
    L = 5 * overlap       #子块的长和宽
    coord_list = []
    geocoord_list = []
    #----------------------计算x，y两个方向可分割的块数-------------#
    if (w-overlap) % (L-overlap) == 0:
        x_num = (w - overlap) / (L - overlap)
    else:
        x_num = math.floor((w-overlap)/(L-overlap))+1 #x方向可切割的块数
    if (h-overlap) % (L-overlap) == 0:
        y_num = (h - overlap) / (L - overlap)
    else:
        y_num = math.floor((h-overlap)/(L-overlap))+1 #y方向可切割的块数

    #---------------------裁剪子区域-----------------------------#
    for j in range(1, y_num + 1):
        for i in range(1, x_num + 1):
            if i == x_num and j != y_num:    #大图像右边缘的块
                w_start=(i-1)*(L-overlap)
                w_end = w
                h_start=(j-1)*(L-overlap)
                h_end=j*(L-overlap)+overlap
            elif i != x_num and j == y_num:  #大图像下边缘的块
                w_start=(i-1)*(L-overlap)
                w_end=i*(L-overlap)+overlap
                h_start=(j-1)*(L-overlap)
                h_end = h
            elif i == x_num and j == y_num:  #大图像右下角的块
                w_start=(i-1)*(L-overlap)
                w_end = w
                h_start=(j-1)*(L-overlap)
                h_end = h
            else:                             #大图像其他块
                w_start=(i-1)*(L-overlap)
                w_end=i*(L-overlap)+overlap
                h_start=(j-1)*(L-overlap)
                h_end = j*(L-overlap)+overlap
            box = (w_start, h_start,w_end, h_end)  #子区域左上角和右下角坐标
            sub_img = img.crop(box)  #裁剪子区域
            if (j - 1) * x_num + i < 10:
                name = '000' + str((j - 1) * x_num + i) + '.jpg'
            elif (j - 1) * x_num + i < 100:
                name = '00' + str((j - 1) * x_num + i) + '.jpg'
            elif (j - 1) * x_num + i < 1000:
                name = '0' + str((j - 1) * x_num + i) + '.jpg'
            else:
                name = str((j - 1) * x_num + i) + '.jpg'
            sub_img.save(os.path.join(save_sub_path, name))  # 保存每块图片
            r_image, coord_list, geocoord_list = yolo.detect_image(img, sub_img, name, i, j, L, overlap, model, map_x, map_y, ct, coord_list, geocoord_list)# 可以不创建文件，但一定要创建文件夹
            # r_image.save(os.path.join(save_cls_path, name))
    print("finish")
    coord_array = np.array(coord_list)
    geocoord_array = np.array(geocoord_list)
    coord_array.shape = -1,4
    geocoord_array.shape = -1,2
    wide = 10
    img = img.convert('RGB')
    draw = ImageDraw.Draw(img)
    for k in range(1, coord_array.shape[0]):
        box1 = coord_array[k, :]
        for m in range(k):
            box2 = coord_array[m, :]
            IOU, b1_area, b2_area = bbox_IOU(box1, box2)
            if IOU > 0.05:
                if b1_area >= b2_area:
                    coord_array[m, :] = 0
                else:
                    coord_array[k, :] = 0
    geocoord_array1 = []
    for k in range(coord_array.shape[0]):
        if sum(coord_array[k, :]) != 0:
            geocoord_array1.append(geocoord_array[k, :])
    geocoord_array1 = np.array(geocoord_array1)
    print(geocoord_array1)

    coord_array1 = []
    for k in range(coord_array.shape[0]):
        if sum(coord_array[k, :]) != 0:
            coord_array1.append(coord_array[k, :])
    coord_array1 = np.array(coord_array1)
    # print(coord_array1)

    for k in range(coord_array1.shape[0]):
        left = coord_array1[k, 0]
        right = coord_array1[k, 1]
        top = coord_array1[k, 2]
        bottom = coord_array1[k, 3]
        for i in range(1,wide+1):
            draw.rectangle((left-i,top-i,right+i,bottom+i), fill =None, outline='yellow')
    img.save(os.path.join(save_det_path, "2_2.jpg"))

    # np.savetxt("pixel_coord.txt", coord_array, fmt='%f', delimiter=',')
    np.savetxt("geo_coord.txt",geocoord_array1, fmt="%s")