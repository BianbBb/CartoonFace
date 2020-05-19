import torch
import os
from torch import nn

class BaseTrainer(object):
    def __init__(self, para, net, optimizer=None, loss=None,):
        self.para = para
        self.logger = self.para.logger
        self.EPOCH = self.para.EPOCH
        self.BATCH_SIZE = self.para.BATCH_SIZE
        self.net = net
        self.device = self.para.device

        torch.cuda.manual_seed(self.para.SEED)
        # torch.cuda.set_device(self.device)
        torch.backends.cudnn.benchmark = True

        self.exp_dir = para.exp_dir
        self.exp_name = para.exp_name
        self.exp_path = os.path.join(self.exp_dir, self.exp_name)

        # if len(self.para.gpu_ids) > 0:
        #     self.net = nn.DataParallel(net, device_ids=self.para.gpu_ids)
        self.net.to(self.device)

        if self.para.resume:  # 从文件中读取模型参数
            self.load_weight()
        self.optimizer_name = optimizer
        self.set_optimizer()

    def load_weight(self):
        try:
            self.net.load_state_dict(torch.load(self.exp_path))
            self.logger.info('Net Parameters Loaded Successfully!')
        except FileNotFoundError:
            self.logger.warning('Can not find feature.pkl !')
        '''
         except torch weights name mismatch：
            log
            bujiazai 
         
        '''


    def set_optimizer(self):
        if self.optimizer_name is 'Adam':
            self.optimizer = torch.optim.Adam(
                self.net.parameters(),
                lr=self.para.lr,)

        elif self.optimizer_name is 'SGD':
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.para.lr,
                momentum=self.para.momentum,
                weight_decay=self.para.weight_decay)

    def train(self):
        raise NotImplementedError

    def val(self):
        raise NotImplementedError

    def get_losses(self):
        raise NotImplementedError

    def run_epoch(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

