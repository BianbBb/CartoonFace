import time
import torch
import numpy as np
import os
import cv2

import sys
sys.path.append("..")

from detect.losses import FocalLoss,BGLoss
from detect.losses import RegL1Loss, RegLoss, NormRegL1Loss
# from .decode import ctdet_decode
from utils.util import _sigmoid
# from utils.debuger import Debugger
# from utils.post_process import ctdet_post_process
from utils.gen_oracle_map import gen_oracle_map
from base_trainer import BaseTrainer
from utils.util import AverageMeter



class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        outputs = self.model(batch['input'])
        # input输入model，得到输出：hm和wh
        loss, loss_stats = self.loss(outputs, batch)
        return outputs, loss, loss_stats

class CtdetLoss(torch.nn.Module):
    def __init__(self, para):
        super(CtdetLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_bg = BGLoss()
        self.crit_reg = RegL1Loss() if para.reg_loss == 'l1' else \
            RegLoss() if para.reg_loss == 'sl1' else None
        self.crit_wh = self.crit_reg  # 或NormRegL1Loss()
        self.para = para

    def forward(self, output, batch):
        hm_loss, wh_loss, off_loss = 0, 0, 0

        output['hm'] = _sigmoid(output['hm'])
        output['hm_b'] = _sigmoid(output['hm_b'])
        hm_loss = self.crit(output['hm'], batch['hm']) ## heatmap loss
        gt_bg = 1 - batch['hm'] * 2
        hm_b_loss = self.crit_bg(output['hm'], gt_bg)

        ## wh loss
        wh_loss = wh_loss + self.crit_reg(output['wh'], batch['reg_mask'], batch['ind'], batch['wh'])

        loss = 1.0 * hm_loss + 1.0 * hm_b_loss + 0.1 * wh_loss

        loss_stats = {'loss': loss, 'hm_loss': hm_loss, 'hm_b_loss': hm_b_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss}
        return loss, loss_stats


class DetTrainer(BaseTrainer):
    def __init__(self, para, net, train_loader, val_loader=None, optimizer=None):
        super(DetTrainer, self).__init__(para, net, optimizer, )
        self.exp_dir = para.exp_dir
        self.exp_name = para.exp_name
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.BEST_VAL_LOSS = None  # 在验证集上的最好结果
        self.VAL_LOSS = None

        self.loss_stats, self.loss = self.get_losses(para)
        self.model_with_loss = ModelWithLoss(net, self.loss)
        self.logger = para.logger
        self.para = para

    def get_losses(self, para):
        # loss_states = ['loss', 'hm_loss', 'wh_loss', 'off_loss']
        loss_states = ['loss', 'hm_loss','hm_b_loss', 'wh_loss']
        loss = CtdetLoss(para)
        return loss_states, loss


    def freeze_layer(self, Freeze_List):
        pass


    def run(self):
        for epoch in range(self.para.EPOCH):
            torch.cuda.empty_cache()
            self.logger.debug('|  Train  Epoch : {} ------------------------  |'.format(epoch))
            self.train()
            self.logger.debug('|  Val  Epoch : {} ------------------------  |'.format(epoch))
            self.val()

            self.logger.info('|Val Loss: {:.4f}'.format(np.mean(self.VAL_LOSS)))
            if self.BEST_VAL_LOSS is None:
                self.BEST_VAL_LOSS = np.mean(self.VAL_LOSS)
                self.save_model()
            else:
                if np.mean(self.VAL_LOSS) <= self.BEST_VAL_LOSS:
                    self.BEST_VAL_LOSS = np.mean(self.VAL_LOSS)
                    self.save_model()


    def save_model(self):
        pkl_save_name = 'sk-{}-{:.3f}.pkl'.format(
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
        results = {}
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        for step, batch in enumerate(data_loader):

            torch.cuda.empty_cache()
            for k in batch:
                batch[k] = batch[k].to(device=self.para.device, non_blocking=True)

            output, loss, loss_stats = self.model_with_loss(batch)
            # batch :dict = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
            loss = loss.mean()

            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            step_time.update(time.time() - t1)
            t1 = time.time()

            for l in avg_loss_stats:
                avg_loss_stats[l].update(loss_stats[l].mean().item(), batch['input'].size(0))

            if step % self.para.log_step == 0:
                self.logger.info('| Step: {:<4d} | Time: {:.2f} | Loss: {:.4f} '
                                 '| hm loss: {:.4f} | hm_b loss: {:.4f} | wh loss: {:.4f}'.format(
                    step, step_time.avg, avg_loss_stats['loss'].avg,
                           avg_loss_stats['hm_loss'].avg,avg_loss_stats['hm_b_loss'].avg,avg_loss_stats['wh_loss'].avg))

        if not is_train:
            self.VAL_LOSS = avg_loss_stats['loss'].avg
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        self.logger.info('| Epoch Time: {:.2f} '.format(time.time()-t0))
        return ret, results

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

