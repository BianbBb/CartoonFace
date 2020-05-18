import numpy as np
import torch
import os
import logging


'''
定义项目中使用的全局变量
网络相关的超参数
'''
class Detection_Parameter():
    def __init__(self):
        # Model Path
        self.resume = False
        # self.exp_dir = 'G:\BBBLib\CartoonFace\exp\\'
        self.exp_dir = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/exp/'
        self.exp_name = 'centernet.pkl'
        self.exp_path = os.path.join(self.exp_dir, self.exp_name)
        # Dataset Setting
        # self.img_dir = 'G:\BBBLib\CartoonFace\data\personai_icartoonface_dettrain\icartoonface_dettrain\\'
        # self.anno_path = 'G:\BBBLib\CartoonFace\data/personai_icartoonface_dettrain/icartoonface_dettrain_short.csv'

        self.img_dir = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/' \
                       'data/personai_icartoonface_dettrain/icartoonface_dettrain/'
        self.anno_path = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/' \
                         'data/personai_icartoonface_dettrain/icartoonface_dettrain.csv'

        self.max_objs = 5  # 一张image中最多目标数量

        self.no_color_aug = False
        self.not_rand_crop = False
        self.shift = 0.1
        self.scale = 0.4
        self.flip = 0.5

        self.input_h = 512
        self.input_w = 512
        self.down_ratio = 4
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio

        self.num_classes = 1
        self.heads = {'hm': self.num_classes,  'wh': 2}
        self.head_conv = 64
        self.num_branchs = 4  # Hourglass的子分支数目

        self.reg_offset = False
        if self.reg_offset:
            self.heads.update({'reg': 2})

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

        self.seed = 2020

        # Network Setting
        self.gpu_ids = [0, 1, 2, 3]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimizer
        self.optimizer = 'SGD'
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-5

        # Loss
        self.reg_loss = 'l1'
        self.eval_oracle_hm = True # Use Ground Truth
        self.eval_oracle_wh = True


        # Train
        self.train_num = 40000  # 在训练数据中选取前40000个作为训练集，其余为验证集
        self.BATCH_SIZE = 4  # 10,16,20...
        self.EPOCH = 200
        self.log_step = 20  # 每隔20step，输出一次训练信息
        self.SEED = 2020



        # Logging
        self.logger = self.get_logger()

        self.logger.debug('------------ Basic Setting ------------')
        self.logger.debug('Device     : {}'.format(self.device))
        self.logger.debug('Image Dir  : {}'.format(self.img_dir))
        self.logger.debug('Ann Path   : {}'.format(self.anno_path))
        self.logger.debug('Image Size : h:{},w:{}'.format(self.input_h,self.input_w))
        self.logger.debug('Exp Path   : {}'.format(self.exp_path))
        self.logger.debug('Heads      : {}'.format(self.heads))
        self.logger.debug('Batch Size : {}'.format(self.BATCH_SIZE))
        self.logger.debug('---------------------------------------')

    def get_logger(self, logger_name='my_logger'):
        logger = logging.getLogger(logger_name)
        logger.setLevel('DEBUG') # 部署时修改为 INFO
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s[%(levelname)s]     %(message)s ")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

if __name__ == '__main__':
    para = Detection_Parameter()








