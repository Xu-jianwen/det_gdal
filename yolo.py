#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import cv2
import numpy as np
import colorsys
import os
import torch
import torch.nn as nn
from nets.yolo4 import YoloBody
import torch.backends.cudnn as cudnn
from PIL import Image,ImageFont, ImageDraw
from torch.autograd import Variable
from utils.utils import non_max_suppression, bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes

from PIL import Image
from scipy import ndimage
import classify as cls

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#--------------------------------------------#
# MBR
def niu_MBR(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  # PIL转cv2，方便灰度化和二值化
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)  # 图像灰度化
    # ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)  # 图像二值化
    # img_binary = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
    ret, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary = cv2.medianBlur(img_binary, 5)
    R = np.zeros((180), dtype='float64')
    for s in range(180):  # 遍历0-179共180个角度
        rotation = ndimage.rotate(img_binary, s, reshape=False).astype(
            'float64')  # 旋转多维array，s表示旋转角，如果reshape为true, 则调整输出形状，以便输入数组完全包含在输出中
        R_i = rotation.sum(axis=0)  # 对array的每一列进行相加
        R_i = sorted(R_i, reverse=True)  # 降序排列
        R[s] = sum(R_i[0:4])  # 取最高的前五个求和，避免离群点的影响
    theta_rot = np.argmax(R)  # 求180个角度中哪个角度对应的R（像素累加值）最大，返回最大值对应的索引,若有相同最大值，返回第一个最大值的索引。
    angle = theta_rot - 90  # 计算需要旋转的方向，由于累加值是垂直统计的，所以相差90度
    binary_PIL = Image.fromarray(img_binary)  # cv2转PIL，方便图像层面的旋转操作
    I_rot = binary_PIL.rotate(angle)  # 图像逆时针旋转固定角度，至水平方向
    bw_rot = np.array(I_rot)  # PIL转数组形式，船只水平的二值化图像
    [x, y] = I_rot.size  # 图像的宽和高
    x_cum = bw_rot.sum(axis=0)  # 对array的每一列进行相加
    y_cum = bw_rot.sum(axis=1)  # 对array的每一行进行相加
    x_ind = np.where(x_cum >= 0.3 * max(x_cum))[0]  # 返回满足条件的索引
    y_ind = np.where(y_cum >= 0.3 * max(y_cum))[0]
    # np.where()[0] 表示行的索引，np.where()[1] 则表示列的索引
    margin = 1  # 设置填充的边框宽度
    # 以下条件语句均是为了防止加margin的裁剪坐标超出图像范围而添加的
    # 计算图像x方向左边裁剪位置
    if min(x_ind) - margin > 0:
        x_min = min(x_ind) - margin
    else:
        x_min = 0
    # 计算图像x方向右边裁剪位置
    if max(x_ind) + margin < x:
        x_max = max(x_ind) + margin
    else:
        x_max = x - 1
    # 计算图像y方向上边裁剪位置
    if min(y_ind) - margin > 0:
        y_min = min(y_ind) - margin
    else:
        y_min = 0
    # 计算图像y方向下边裁剪位置
    if max(y_ind) + margin < y:
        y_max = max(y_ind) + margin
    else:
        y_max = y - 1
    img_rot = img.rotate(angle)  # 将原始图像旋转至水平方向
    box = (x_min, y_min, x_max, y_max)  # 子区域左上角和右下角坐标
    img_MBR = img_rot.crop(box)  # 裁剪子区域
    return img_MBR
# ---------------------------------------------------------------------------------------#


class YOLO(object):
    _defaults = {
        "model_path": 'logs/Epoch100-Total_Loss2.6283-Val_Loss6.0589.pth',  # first model
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/ship_class.txt',
        "model_image_size" : (416, 416, 3),
        "confidence": 0.2,
        "iou" : 0.3,
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    #---------------------------------------------------#
    #   获得所有的先验框
    #---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape([-1, 3, 2])[::-1,:,:]

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self):
        
        self.net = YoloBody(len(self.anchors[0]),len(self.class_names)).eval()

        # 加快模型训练的效率
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        
        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
    
        print('Finished!')

        self.yolo_decodes = []
        for i in range(3):
            self.yolo_decodes.append(DecodeBox(self.anchors[i], len(self.class_names),  (self.model_image_size[1], self.model_image_size[0])))


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, img, image, image_nobox, name, m, n, L, overlap, model):
        # f = open('output.txt',mode='w')
        image_shape = np.array(np.shape(image)[0:2])
        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))    # ?
        photo = np.array(crop_img,dtype = np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            
        output_list = []

        for i in range(3):

            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                                conf_thres=self.confidence,
                                                nms_thres=self.iou)
        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        thickness = (np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0]


        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            top1 = top+(n-1)*(L-overlap)
            left1 = left+(m-1)*(L-overlap)
            bottom1 = bottom+(n-1)*(L-overlap)
            right1 = right+(m-1)*(L-overlap)
            #------------------------------------------------------------------------------------------#
                                                    #MBR
            ROOT_DIR = os.getcwd()
            xmlfilepath = os.path.join(ROOT_DIR, r"Annotations/")
            saveBasePath = os.path.join(ROOT_DIR, r"ImageSets/Main/")

            pathsub_img = os.path.join(ROOT_DIR,'output/crop_ship/')#截取的检测到的船只
            pathsub_MBR = os.path.join(ROOT_DIR,'output/MBR_ship/')#船只最小外接矩形
            scale = 0.3
            x_min = max(round(left - scale * (right - left)) + (m-1)*(L-overlap), 0)
            x_max = min(round(right + scale * (right - left)) + (m-1)*(L-overlap), np.shape(img)[1])
            y_min = max(round(top - scale * (bottom - top)) + (n-1)*(L-overlap), 0)
            y_max = min(round(bottom + scale * (bottom - top)) + (n-1)*(L-overlap), np.shape(img)[0])
            box = (x_min, y_min, x_max, y_max)
            sub_img = img.crop(box)
            sub_img.save(os.path.join(pathsub_img, name[0:4] + '_' + str(i) + '.jpg'))  # gcg
            MBR_image = niu_MBR(sub_img)
            wide, high = MBR_image.size


            MBR_image.save(os.path.join(pathsub_MBR, name[0:4] + '_' + str(i) + '.jpg'))  # gcg


            #---------------------------------------classification------------------------------------#
            image_dim_len = len(np.array(MBR_image).shape)
            if image_dim_len ==3:
                src_RGB = MBR_image
            else:
                src_RGB = np.zeros((high, wide,3))
                src_RGB[:, :, 0] = MBR_image
                src_RGB[:, :, 1] = MBR_image
                src_RGB[:, :, 2] = MBR_image
            src_RGB = Image.fromarray(np.uint8(src_RGB)).convert('RGB')
            softmax, pred_label = cls.test(src_RGB, model)
            label_cls = '{:s} {:.4f}'.format(pred_label,softmax)
            output_log = '{:s} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(pred_label,softmax, top1, left1, bottom1, right1)
            #------------------------------------------------------------------------------------------#
            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            image = image.convert('RGB')
            draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)
            label_cls_size = draw.textsize(label_cls, font)   #niu
            label_cls = label_cls.encode('utf-8')        #niu
            # label = label.encode('utf-8')
            print(output_log)

            #---------------------------niu-----------------------------------#
            if top - label_cls_size[1] >= 0:
                text_origin = np.array([left, top - label_cls_size[1]])
            else:
                text_origin = np.array([left, top + 1])
            #-----------------------------------------------------------------#
            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            # else:
            #     text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            #---------------------------niu-----------------------------------#
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_cls_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            draw.text(text_origin, str(label_cls, 'UTF-8'), fill=(0, 0, 0), font=font)
            # # ----------------------------------------------------------------#
            # draw.rectangle(
            #     [tuple(text_origin), tuple(text_origin + label_size)],
            #     fill=self.colors[self.class_names.index(predicted_class)])
            # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image
