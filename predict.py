from yolo import YOLO
from PIL import Image
import os
import math
import cv2
import torch
import sys



Image.MAX_IMAGE_PIXELS = None
yolo = YOLO()
pre_log = os.getcwd()
path_pre = os.path.join(pre_log,r"resnet18_sar_model.pth")#子图像保存路径
print(path_pre)
model = torch.load(path_pre)
model.cuda()
model.eval()



#---------------------print-to-txt----------------#
open('output_log.txt', 'w').close()
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('output_log.txt')

#-----------------------------------CROP-------------------------#
root_dir = os.getcwd()
path = os.path.join(root_dir, r'input/')#原始图片路径
im_names = os.listdir(path)
im_names.sort()
for im_name in im_names:
    src_path = os.path.join(path,im_name)
    save_sub_path = os.path.join(root_dir,r'output/crop_result/')#子图像保存路径
    out_path= os.path.join(root_dir,r'output/cls_result/')#最终分类结果
    img = Image.open(src_path)#打开读取图像
    w, h= img.size #获得图像尺寸
    L = 1100       #子块的长和宽
    overlap = 150  #重叠的像素个数
    ###############计算x，y两个方向可分割的块数#########################
    if (w-overlap)%(L-overlap) == 0:
        x_num = (w - overlap) / (L - overlap)
    else:
        x_num = math.floor((w-overlap)/(L-overlap))+1 #x方向可切割的块数
    if (h-overlap)%(L-overlap) == 0:
        y_num = (h - overlap) / (L - overlap)
    else:
        y_num = math.floor((h-overlap)/(L-overlap))+1 #y方向可切割的块数

    #######################裁剪子区域###############################
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
            sub_img1 = sub_img
            if (j - 1) * x_num + i < 10:
                name = '000' + str((j - 1) * x_num + i) + '.jpg'
            elif (j - 1) * x_num + i < 100:
                name = '00' + str((j - 1) * x_num + i) + '.jpg'
            elif (j - 1) * x_num + i < 1000:
                name = '0' + str((j - 1) * x_num + i) + '.jpg'
            else:
                name = str((j - 1) * x_num + i) + '.jpg'
            sub_img.save(os.path.join(save_sub_path, name))  # 保存每块图片
            r_image = yolo.detect_image(img, sub_img, sub_img1, name, i, j, L, overlap, model) # 可以不创建文件，但一定要创建文件夹
            r_image.save(os.path.join(out_path, name))
