import torch.utils.data as DATA
import numpy as np
import torch
import json
import math
import cv2
import os
import csv
import sys
sys.path.append("..")

from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg


# 用于目标检测的数据
# 适用于CenterNet

class DetDataset(DATA.Dataset):
    def __init__(self, para, ):
        '''

        # 读取csv文件
        # 一张图片上可能会有多个box标注，需要都加在一个列表内

        box_dict =
        {'personai_icartoonface_dettrain_00001.jpg': [[246, 141, 401, 296]], #[xmin,ymin,xmax,ymax]
        'personai_icartoonface_dettrain_00005.jpg': [[323, 219, 391, 311]],
        'personai_icartoonface_dettrain_00010.jpg': [[205, 149, 258, 223], [536, 286, 565, 334], [379, 389, 408, 423], [336, 407, 360, 426]],
        'personai_icartoonface_dettrain_00020.jpg': [[102, 330, 315, 446]],
        'personai_icartoonface_dettrain_00027.jpg': [[184, 301, 313, 426]],
        'personai_icartoonface_dettrain_00052.jpg': [[127, 184, 191, 229], [163, 69, 269, 148], [268, 46, 336, 147]],
        ...
        }

        '''

        self.para = para
        self.img_dir =  self.para.img_dir
        self.anno_path =  self.para.anno_path
        self.max_objs =  self.para.max_objs

        self.box_dict ={}

        with open(self.anno_path, 'r') as f:
            lines = csv.reader(f)
            for line in lines:  # img_path,xmin.ymin,xmax,ymax
                img_name = line[0]
                if not (os.path.isfile(os.path.join(self.img_dir , img_name))):
                    continue
                box = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]  # xmin,ymin,xmax,ymax

                if img_name not in self.box_dict.keys():
                    self.box_dict[img_name] = [box]
                else:
                    self.box_dict[img_name].append(box)

                # aa = box
                # clip = img[aa[1]:aa[3], aa[0]:aa[2]]
                # clip = cv2.resize(clip, (512, 512))
                # cv2.imshow('img', img)
                # cv2.imshow('clip', clip)
                # cv2.waitKey(0)

    def __getitem__(self, index):
        file_name = list(self.box_dict.keys())[index]
        img_path = os.path.join(self.img_dir, file_name)
        anns = self.box_dict[file_name]
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  ## 获取中心点

        # 将图片仿射变换为512,512
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = self.para.input_h, self.para.input_w

        # 数据增强
        flipped = False # 翻转增强的flag
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

        # 生成heatmap
        output_h = input_h // self.para.down_ratio  # 输出512//4=128
        output_w = input_w // self.para.down_ratio
        num_classes = self.para.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        # 有80个类别，最多回归100个中心点
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)  # heatmap(80,128,128)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)  # 中心点宽高(100*2)
        # dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)  # 返回2*128*128
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)  # 记录下采样带来的误差,返回100*2的小数
        ind = np.zeros((self.max_objs), dtype=np.int64)  # 返回100个ind
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)  # 返回100个 回归mask
        ## max_obj 为图中最多有多少个中心点 ，设置为20

        draw_gaussian = draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            bbox = np.array(anns[k])

            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                # ct为目标中心点的坐标
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[0], ct_int, radius)
                # cv2.imwrite(path, hm[0]*255) # 数据预处理可以保存全部的guass，加快后续的训练过程

                wh[k] = 1. * w, 1. * h  # 目标矩形框的宽高——目标尺寸损失
                ind[k] = ct_int[1] * output_w + ct_int[0]  # 目标中心点在128×128特征图中的索引
                reg[k] = ct - ct_int  # off Loss, # ct 即 center point reg是偏置回归数组，存放每个中心店的偏置值 k是当前图中第k个目标
                # 实际例子为
                # [98.97667 2.3566666] - [98  2] = [0.97667, 0.3566666]
                reg_mask[k] = 1  # 有目标的位置的mask


                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, 0])

        # cv2.imwrite(path,hm[0]*255)

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}

        # if self.para.reg_offset: # 是否计算下采样偏差
        #     ret.update({'reg': reg})


        '''
        'input': inp,                  图像
        'hm': hm,                      中心点
        'reg_mask': reg_mask,          ?最大5个目标，记录对应的五个输出是否都为目标
        'ind': ind,                    中心在128*128(即下采样4倍后)上的索引
        'wh': wh                       目标框的宽与高
        'reg': reg                     offset 中心点的误差
       
        '''
        return ret

    def __len__(self):
        return len(self.box_dict.keys())

    def _get_border(self, border, size):
        # border 128  pic_len w or h
        i = 1
        while size - border // i <= border // i:
            # 如果图像宽高小于 boder*2，i增大，返回128 // i
            # 正常返回128，图像小于256，则返回64
            i *= 2
        return border // i




if __name__ == '__main__':
    import config as cfg
    import logging
    LOG_FORMAT = "%(asctime)s[%(levelname)s]     %(message)s "
    DATE_FORMAT = '%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, )

    parameter = cfg.Detection_Parameter()
    dataset = DetDataset(parameter)
    train_loader = DATA.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    for step, data in enumerate(train_loader):
        logging.debug(step)

        print(data['reg_mask'])
        ind = (data['ind']).numpy()[0]
        wh = (data['wh']).numpy()[0]

        input = data['input']
        input = input.numpy()[0]
        input = input.transpose([1,2,0])

        hm = data['hm']
        hm = hm.numpy()[0][0]
        for i in range(len(ind)):
            y = ind[i]//128
            x = ind[i] % 128
            w,h  = wh[i]
            hm = cv2.rectangle(hm, (x-int(w/2), y-int(h/2)), (x+int(w/2), y+int(h/2)), (255, 255, 255), 1)
        hm = cv2.resize(hm,(512,512))

        cv2.imshow('input', input)
        cv2.imshow('hm', hm)
        cv2.waitKey(0)




