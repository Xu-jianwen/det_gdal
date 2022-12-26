from __future__ import print_function
import torch
import torch.nn.functional as F
import os
from PIL import Image
from torchvision import transforms

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# no_cuda = False
# cuda = not no_cuda and torch.cuda.is_available()

def test(data,model):
    # model.eval() # 设置为test模式

    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = transform(data).unsqueeze(0)
    data = data.cuda()

    with torch.no_grad():
        ft, output = model(data)
        Softmax = F.softmax(output,dim=1)
        soft = Softmax.cpu().numpy()
        if max(soft[0]) < 0.6:
            pred = 3
        else:
            pred = torch.max(Softmax,1)[1]  # get the index of the max log-probability
        label = ['cargo', 'container', 'tanker', 'others']
        softmax = max(soft[0])
        pred_label = label[pred]
    return softmax, pred_label

# if __name__ == '__main__':
#     data = Image.open('Dataset/SAR_data_JKF/test_data/test/contianer/container_41_3530.jpg')
#     model = torch.load('resnet_sar_model.pth')
#     model.cuda()
#     softmax, pred_label = test(data,model)
#     print('分类得分:{:.4f}\t 标签：{:s}'.format(softmax,pred_label))




