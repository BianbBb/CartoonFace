import time
import torch
from torch import nn
import numpy as np
import os
from detect.losses import FocalLoss
from detect.losses import RegL1Loss, RegLoss, NormRegL1Loss
# from .decode import ctdet_decode
from utils.util import _sigmoid
# from utils.debuger import Debugger
# from utils.post_process import ctdet_post_process
from utils.gen_oracle_map import gen_oracle_map
from recognition.base_trainer import BaseTrainer
from utils.util import AverageMeter
from recognition.losses import convert_label_to_similarity, CircleLoss


class RecTrainer(BaseTrainer):
    def __init__(self, para, net, train_loader, val_loader=None, optimizer=None):
        super(RecTrainer, self).__init__(para, net, optimizer, )
        self.exp_dir = para.exp_dir
        self.exp_name = para.exp_name
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.BEST_VAL_LOSS = None  # 在验证集上的最好结果
        self.VAL_LOSS = None
        self.loss = CircleLoss(m=0.25, gamma=80)

        self.logger = para.logger
        self.para = para

    def freeze_layer(self, Freeze_List):
        pass

    def run(self):
        for epoch in range(self.para.EPOCH):
            torch.cuda.empty_cache()
            self.logger.debug('|  Train  Epoch : {} ------------------------  |'.format(epoch))
            self.train()
            self.logger.debug('|  Val  Epoch : {} ------------------------  |'.format(epoch))
            self.val()

            self.logger.info('|Val Loss: {:.4f}'.format(self.VAL_LOSS))
            if self.BEST_VAL_LOSS is None:
                self.BEST_VAL_LOSS = self.VAL_LOSS
                self.save_model()
            else:
                if self.VAL_LOSS <= self.BEST_VAL_LOSS:
                    self.BEST_VAL_LOSS = self.VAL_LOSS
                    self.save_model()


    def save_model(self):
        pkl_save_name = 'resnest -{}-{:.3f}.pkl'.format(
            time.strftime("%m%d-%H%M", time.localtime()), self.BEST_VAL_LOSS)
        pkl_save_path = os.path.join(self.exp_dir, pkl_save_name)
        torch.save(self.net.state_dict(), pkl_save_path)

    def train(self):
        return self.run_epoch(self.train_loader, is_train=True)

    @torch.no_grad()
    def val(self):
        return self.run_epoch(self.val_loader, is_train=False)
        # self.BEST_VAL_LOSS = None
        # self.net.eval()
        # net.eval()
        #
        # VAL_LOSS = []
        # for step, (images, instances) in enumerate(val_loader):
        #     torch.cuda.empty_cache()
        #     with torch.no_grad():
        #         anchor = images[0]
        #         pos = images[1]
        #         neg = images[2]
        #
        #         if _CUDA is True:
        #             anchor = anchor.cuda()
        #             pos = pos.cuda()
        #             neg = neg.cuda()
        #
        #         f_anchor = net(anchor)  # （b,1000）
        #         f_pos = net(pos)
        #         f_neg = net(neg)
        #
        #         # 将feature,label在batch维度上拼接
        #         features = nn.functional.normalize(torch.cat((f_anchor, f_pos, f_neg), 0))
        #         labels = torch.cat((instances[0], instances[1], instances[2]), 0)
        #
        #         inp_sp, inp_sn = convert_label_to_similarity(features, labels)
        #         val_loss = _loss(inp_sp, inp_sn)
        #
        #     val_loss_np = val_loss.data.cpu().numpy()
        #     VAL_LOSS.append(val_loss_np)

    def run_epoch(self, data_loader, is_train=True, epoch=0 ):
        if is_train:
            self.net.train()
        else:
            self.net.eval()

        t0 = time.time()  # epoch timer
        t1 = time.time()  # step timer
        step_time = AverageMeter()
        epoch_loss = AverageMeter()

        for step, (images, instances) in enumerate(data_loader):
            torch.cuda.empty_cache()
            anchor = images[0]
            pos = images[1]
            neg = images[2]

            anchor = anchor.to(device=self.para.device, non_blocking=True)
            pos = pos.to(device=self.para.device, non_blocking=True)
            neg = neg.to(device=self.para.device, non_blocking=True)

            f_anchor = self.net(anchor)   # (b,512)
            f_pos = self.net(pos)
            f_neg = self.net(neg)

            # 将feature,label在batch维度上拼接
            features = nn.functional.normalize(torch.cat((f_anchor, f_pos, f_neg), 0))

            labels = torch.cat((instances[0], instances[1], instances[2]), 0)

            inp_sp, inp_sn = convert_label_to_similarity(features, labels)
            loss = self.loss(inp_sp, inp_sn)

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            step_time.update(time.time() - t1)
            t1 = time.time()

            epoch_loss.update(loss)
            if step % self.para.log_step == 0:
                self.logger.info('| Step: {:<4d} | Time: {:.2f} | Loss: {:.4f}'.format(
                    step, step_time.avg, epoch_loss.avg))

        if not is_train:
            self.VAL_LOSS = epoch_loss.avg

        self.logger.info('| Epoch Time: {:.2f} '.format(time.time()-t0))


'''  
Circle Loss 
Net process
# anchor = images[0]
        # pos = images[1]
        # neg = images[2]
        #
        # if _CUDA is True:
        #     anchor = anchor.cuda()
        #     pos = pos.cuda()
        #     neg = neg.cuda()
        #
        # f_anchor = net(anchor)  # （b,1000）
        # f_pos = net(pos)
        # f_neg = net(neg)
        #
        # # 将feature,label在batch维度上拼接
        # features = nn.functional.normalize(torch.cat((f_anchor, f_pos, f_neg), 0))
        # labels = torch.cat((instances[0], instances[1], instances[2]), 0)
        #
        # inp_sp, inp_sn = convert_label_to_similarity(features, labels)
        # loss = _loss(inp_sp, inp_sn)
        # loss.backward()
        # optimizer.step()
        #
        # if step % 5 == 0:  # tianchi:200
        #     if _CUDA is True:
        #         logging.debug('Epoch:{:<3d}   Step:{:<4d}  | Time:{:.2f} | Train Loss:{:.4f}'.format(
        #             epoch, step, time.time() - t1, loss.data.cpu().numpy()))
        #
        #     else:
        #         logging.debug(
        #             'Epoch:{:d}   Step:{:d}   |  train loss:{:.4f}'.format(epoch, step, loss.data.numpy()))
        #     t1 = time.time()
'''

