import torch
import torch.nn as nn
import os
import cv2
import numpy as np
import time
import json

import sys
sys.path.append("..")
import config as cfg
from utils.util import _gather_feat, _transpose_and_gather_feat
from backbone.Hourglass.large_hourglass import HourglassNet
from utils.image import get_affine_transform, transform_preds


def _nms(heat, kernel=3):
    # 8-近邻极大值点 ，极大值点处的heat被保留，其余点为0
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()  # cat 代表类别数目，当前的task为1
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)  # 在每张heatmap上选k个最大值

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # 在每个图片中选k个最大值 ，用于选择类别 ，cat为1时下面代码没有实际意义
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()

    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class Tester(object):
    def __init__(self, para, ):
        self.para = para
        self.logger = para.logger
        self.device = para.device
        self.net = None
        self.test_dir = para.test_dir

    def run(self):
        self.logger.debug('------ Load Network ------')
        self.net = HourglassNet(self.para.heads, num_stacks=1, num_branchs=self.para.num_branchs)
        self.load_weight()
        self.net.eval()

        self.logger.debug('------ Test ------')
        for file_name in os.listdir(self.test_dir):
            img_path = os.path.join(self.test_dir, file_name)
            img = cv2.imread(img_path)
            inputs, meta = self.pre_process(img)
            # meta:c,  s,  out_height:512/4  , out_width:512/4
            inputs = inputs.to(self.para.device)

            # torch.cuda.synchronize()  # 需要计算test时间时使用
            outputs, dets = self.get_output(self.net, inputs)[-1]
            # outputs : 网络的输出 hm:1*w*h  wh:2*w*h
            # dets : [bboxes, scores, clses]

            print(dets)  ##
            dets = self.post_process(dets, meta)
            print(dets)  ##
            results = self.merge_outputs(dets)  # bboxes
            print(results)  ##

            self.show_sample(img, results)
            self.create_json()


    def load_weight(self):
        try:
            if not torch.cuda.is_available():
                self.net.load_state_dict(torch.load(self.para.exp_path, map_location=lambda storage, loc: storage))
            else:
                self.net.load_state_dict(torch.load(self.para.exp_path))
                self.net.to(self.para.device)
            self.logger.info('Net Parameters Loaded Successfully!')
        except FileNotFoundError:
            self.logger.warning('Can not find feature.pkl !')
            return False

    '''
    def get_input(self,img_path, para):
        # 按train进行归一化，修改
        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        h_ratio = height / para.input_h
        w_ratio = width / para.input_w

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)  ## 获取中心点

        # 将图片仿射变换为512,512
        s = max(img.shape[0], img.shape[1]) * 1.0
        input_h, input_w = para.input_h, para.input_w

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)  # 仿射变换
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - para.mean) / para.std

        # cv2.imwrite('./test/'+img_path[-9 :-4]+'_inp.jpg', inp)  
        # print('./test/'+img_path[-9 :-4]+'_inp.jpg')

        inp = inp.transpose(2, 0, 1)

        return inp, h_ratio, w_ratio
    '''

    def pre_process(self, image):
        height, width = image.shape[0:2]

        inp_height, inp_width = self.para.input_h, self.para.input_w
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        # else:
        #     inp_height = (new_height | self.opt.pad) + 1
        #     inp_width = (new_width | self.opt.pad) + 1
        #     c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        #     s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(image, (width, height))
        inp_image = cv2.warpAffine(
            resized_image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.para.mean) / self.para.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        # if self.opt.flip_test:
        #     images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
        images = torch.from_numpy(images)
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.para.down_ratio,
                'out_width': inp_width // self.para.down_ratio}
        return images, meta

    @torch.no_grad()
    def get_output(self, net, inputs):
        torch.cuda.empty_cache()
        inputs = torch.unsqueeze(inputs, dim=0).float()  # inputs的batch_size = 1时
        output = net(inputs)

        hm = output['hm'].sigmoid_()
        wh = output['wh']

        # 将hm,wh转换为bbox
        dets = self.ctdet_decode(hm, wh,  K=self.para.K)

        return output, dets

    def ctdet_decode(self, heat, wh, reg=None, K=40, ):
        batch, cat, height, width = heat.size()
        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        heat = _nms(heat)

        scores, inds, clses, ys, xs = _topk(heat, K=K)
        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        clses = clses.view(batch, K, 1).float() ##
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)
        # detections = torch.cat([bboxes, scores], dim=2)
        return detections

    def post_process(self, dets, meta, scale=1):
        num_classes = 1
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = self.ctdet_post_process(dets.copy(), [meta['c']], [meta['s']],meta['out_height'], meta['out_width'], num_classes)
        for j in range(1, num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
            dets[0][j][:, :4] /= scale
        return dets[0]

    def ctdet_post_process(self, dets, c, s, h, w, num_classes):
        # dets: batch x max_dets x dim
        # return 1-based class det dict
        ret = []
        for i in range(dets.shape[0]):
            top_preds = {}
            dets[i, :, :2] = transform_preds(
                dets[i, :, 0:2], c[i], s[i], (w, h))
            dets[i, :, 2:4] = transform_preds(
                dets[i, :, 2:4], c[i], s[i], (w, h))
            classes = dets[i, :, -1]
            for j in range(num_classes):
                inds = (classes == j)
                top_preds[j + 1] = np.concatenate([
                    dets[i, inds, :4].astype(np.float32),
                    dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
            ret.append(top_preds)
        return ret

    def merge_outputs(self, detections):
        num_classes = 1
        results = {}
        for j in range(1, num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            '''
            # if len(self.scales) > 1 or self.para.nms:
            if self.para.nms:
                soft_nms(results[j], Nt=0.5, method=2)
            '''

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, num_classes + 1)])
        if len(scores) > self.para.max_per_image:
            kth = len(scores) - self.para.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def show_sample(self, img , results):
        cv2.imshow('img',img)
        cv2.waitKey(0)
        print('---------show sample---result---')
        print(results)
        time.sleep(20000)

    def create_json(self):
        print('FFINISH')  ##
        pass


if __name__ == '__main__':
    parameter = cfg.Detection_Parameter()
    tester = Tester(parameter)
    tester.run()
