from PIL import Image
from torchvision import transforms
import logging
import torch.utils.data as data
import os
import json

import random
import cv2
import numpy as np
import torch

import sys
sys.path.append("..")
import config as cfg
from utils.image import get_affine_transform
from utils.image import flip, color_aug


class RecDataset(data.Dataset):
    def __init__(self, para, flag='train', train_num=None):

        self.para = para
        self.infos = {}
        '''
        infos = 
        {'00000':{
                '0000000':{path:'personai_icartoonface_rectrain_00000/personai_icartoonface_rectrain_00000_0000000.jpg',box:[x_min,y_min,x_max,y_max],}
                '0000001':{path:''  ,   box:[x_min,y_min,x_max,y_max]  }
                ...
                }
        
        '00001':{  }
        ... 
        }
        '''

        with open(para.anno_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line_info = line.strip().split()
                path_info = line_info[0].split('_')
                img_id = path_info[6]  # img 类别
                img_idx = path_info[7][:-4]  # img 的序号
                img_info = {}
                img_info['path'] = line_info[0]
                img_info['box'] = [int(v) for v in line_info[1:5]]
                if img_id not in list(self.infos.keys()):
                    self.infos[img_id] = {}
                self.infos[img_id][img_idx] = img_info

        if train_num:
            if flag is 'train':
                train_infos_dict = {}
                for i, (k, v) in enumerate(self.infos.items()):
                    if i == train_num:
                        break
                    train_infos_dict[k] = v
                self.infos = train_infos_dict
            else:
                val_infos_dict = {}
                for i, (k, v) in enumerate(self.infos.items()):
                    if i < train_num:
                        continue

                    val_infos_dict[k] = v
                self.infos = val_infos_dict

        # self.transform = transforms.Compose([  # 图像变换
        #     transforms.Resize((512,512)),  # 调整尺寸
        #     # transforms.RandomCrop(224),
        #     # transforms.RandomResizedCrop(224),  # 随机裁剪,
        #     transforms.RandomHorizontalFlip(),  # 随机水平翻转
        #     # transforms.ToTensor(),  # chw
        #     #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        # ])
        #

        self.transform = self.my_transform

    def my_transform(self, img):
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  ## 获取中心点

        # 将图片仿射变换为512,512
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.para.input_h, self.para.input_w

        # 数据增强
        flipped = False  # 翻转增强的flag
        if not self.para.not_rand_crop:
            s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))  # 随机尺度
            w_border = self._get_border(128, img.shape[1])  # 图像的w<256,为64；w>256,为128 仿射变换需要应用
            h_border = self._get_border(128, img.shape[0])
            c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
            c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        else:
            sf = self.para.scale
            cf = self.para.shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        if np.random.random() < self.para.flip:
            flipped = True
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1  # 随机裁剪

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)  # 仿射变换
        inp = (inp.astype(np.float32) / 255.)
        if not self.para.no_color_aug:
            color_aug(self.para._data_rng, inp, self.para._eig_val, self.para._eig_vec)
        inp = (inp - self.para.mean) / self.para.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def Clip(self,image,box):
        x_min,y_min,x_max,y_max = box
        return image[y_min:y_max, x_min:x_max, :]

    def _get_border(self, border, size):
        # border 128  pic_len w or h
        i = 1
        while size - border // i <= border // i:
            # 如果图像宽高小于 boder*2，i增大，返回128 // i
            # 正常返回128，图像小于256，则返回64
            i *= 2
        return border // i

    def GET_PIC(self, label):
        img_path = ''
        while not os.path.isfile(img_path):
            id_info = self.infos[label]
            img_index = random.randint(0, len(id_info) - 1)
            img_index = list(id_info.keys())[img_index]
            img_info = id_info[img_index]
            img_path = os.path.join(self.para.img_dir, img_info['path'])
            box = [int(v) for v in img_info['box']]

        img = self.loader(img_path)
        img = self.Clip(img, box)
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        pos_key = list(self.infos.keys())[index]  # '00005'
        neg_index = random.randint(0, len(self.infos)-1)
        neg_key = list(self.infos.keys())[neg_index]

        anchor = self.GET_PIC(pos_key)
        pos = self.GET_PIC(pos_key)
        neg = self.GET_PIC(neg_key)

        images = [anchor, pos, neg]
        keys = [pos_key, pos_key, neg_key]
        labels = [int(l) for l in keys]
        return (images,labels)

    def __len__(self):
        return len(self.infos)

    def Clip(self,image,box):
        x_min,y_min,x_max,y_max = box
        return image[y_min:y_max, x_min:x_max, :]

    def loader(self, path, reduce=False):

        img = cv2.imread(path)
        # if reduce == True:
        #     H,W,_ = np.shape(img)

        # img = Image.open(path).convert('RGB')
        # if reduce == True:
        #     W, H = img.size
        #     img = img.resize((int(W / 2), int(H / 2)), Image.BILINEAR)
        return img






if __name__ == '__main__':
    parameter = cfg.Detection_Parameter()
    para = parameter

    # import os
    # from PIL import Image
    #

    #
    # with open(os.path.join(download_path, 'icartoonface_rectrain_det.txt'), 'r', encoding='utf-8') as f:
    #     first_name = None
    #     for line in f.readlines():
    #         line_info = line.strip().split()
    #         imgpath = os.path.join(download_path, 'icartoonface_rectrain', line_info[0])
    #         box = [int(v) for v in line_info[1:5]]
    #         try:
    #             img = Image.open(imgpath)
    #             cropped = img.crop(box)
    #             path_info = line_info[0].split('_')
    #             img_path = path_info[7]
    #             save_img_dir = os.path.join(train_save_path, path_info[6])
    #             if first_name != path_info[6]:
    #                 first_name = path_info[6]
    #                 save_img_dir = os.path.join(val_save_path, path_info[6])
    #             if not os.path.isdir(save_img_dir):
    #                 os.makedirs(save_img_dir)
    #             cropped.save(os.path.join(save_img_dir, img_path))
    #
    #         except FileNotFoundError:
    #             pass
    #

    train_num = 500
    Train_Data = RecDataset(para,flag='train')
    train_loader = data.DataLoader(dataset=Train_Data, batch_size=2, shuffle=True, drop_last=True)

    # Val_Data = RecDataset(DATA_Path,flag='val', train_num=train_num)
    # val_loader = data.DataLoader(dataset=Val_Data, batch_size=2, shuffle=False)

    def Torch_to_CV2(image):
        image = image.numpy()  # 这里index表示anchor/pos/neg
        image = image[0].transpose([1, 2, 0])  # CHW - >HWC #这里index表示batch
        image = image[..., ::-1]
        return image

    for step, (images, label) in enumerate(train_loader):
        print(step)

        anchor = images[0]
        pos = images[1]
        neg = images[2]

        print(type(images))
        print(np.shape(anchor))
        print(label)
        print(type(label))
        input = anchor
        input = Torch_to_CV2(input)


        cv2.imshow('input', input)

        cv2.waitKey(0)

    # for step, images in enumerate(val_loader):
    #     logging.debug(step)


    # get item 会返回 pos(8,3,w,h) neg1(8,3,w,h) neg2(8,3,w,h)

        # anchor = Totch_to_CV2(images[0])
        # pos = Totch_to_CV2(images[1])
        # neg = Totch_to_CV2(images[2])
        #
        # cv2.imshow('anchor',anchor)
        # cv2.imshow('pos',pos)
        # cv2.imshow('neg', neg)
        # cv2.waitKey(0)

    # 训练时，需要再将img送入cuda





