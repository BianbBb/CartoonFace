import numpy as np
import torch
import os
import logging
import platform


'''
定义项目中使用的全局变量
网络相关的超参数
'''

class Base_parameter():
    def __init__(self):
        # 训练后的模型参数
        self.exp_dir = None
        self.exp_path = None
        self.logger = self.get_logger()

        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

        self.SEED = 2020

        # Network Setting
        self.gpu_ids = [0]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.platform = None
        self.set_plat()

    def set_plat(self):
        if platform.system() == 'Windows':
            self.platfrom = 'Windows'
        elif platform.system() == 'Linux':
            self.platfrom = 'Linux'


    def get_logger(self, logger_name='my_logger'):
        logger = logging.getLogger(logger_name)
        logger.setLevel('DEBUG') # 部署时修改为 INFO
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s[%(levelname)s]     %(message)s ")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

class Detection_Parameter(Base_parameter):
    def __init__(self):
        super(Detection_Parameter,self).__init__()
        # Model Path
        self.resume = False
        self.exp_name = 'centernet-0521-1723-1.559.pkl'

        # Dataset
        self.img_dir = None
        self.anno_path = None
        self.test_dir = None
        self.set_path()  # 可选：lab, my, tc # 不填写时自动判断lab 或 my

        # Dataset Setting
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

        # NetWork Setting
        self.model = 'Hourglass'# 'Hourglass'

        self.num_classes = 1
        self.heads = {'hm': self.num_classes,  'wh': 2}
        self.head_conv = 64
        self.num_branchs = 4  # Hourglass的子分支数目
        self.dims = [256, 256,256, 512, 1024]

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



        # Optimizer
        self.optimizer = 'SGD'
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-5

        # Loss
        self.reg_loss = 'l1'
        self.eval_oracle_hm = True  # Use Ground Truth
        self.eval_oracle_wh = True

        # Det_Train
        self.train_num = 40000  # 在训练数据中选取前40000个作为训练集，其余为验证集
        self.BATCH_SIZE = 16  # 4,8,12,16,20...
        self.EPOCH = 200
        self.log_step = 50  # 每隔50step，输出一次训练信息

        # Logging
        self.logger.debug('------------ Basic Setting ------------')
        self.logger.debug('Platform   : {}'.format(self.platfrom))
        self.logger.debug('Device     : {}'.format(self.device))
        self.logger.debug('Image Dir  : {}'.format(self.img_dir))
        self.logger.debug('Anno Path   : {}'.format(self.anno_path))
        self.logger.debug('Image Size : h:{},w:{}'.format(self.input_h,self.input_w))
        self.logger.debug('Exp Path   : {}'.format(self.exp_path))
        self.logger.debug('Heads      : {}'.format(self.heads))
        self.logger.debug('Batch Size : {}'.format(self.BATCH_SIZE))
        self.logger.debug('---------------------------------------')

        # Test
        self.K = 100  # 在heatmap中选取的极大值点的个数
        self.max_per_image = 20 # 20
        self.nms = True
        self.result_csv = '/home/byh/CartoonFace/result/result.csv'

    def set_path(self, plat=None):
        if plat is None:
            if self.platfrom ==  'Windows':
                plat = 'my'
            elif self.platfrom ==  'Linux':
                plat = 'lab'

        if plat is 'lab':
            self.exp_dir = '/home/byh/CartoonFace/exp/'
            self.exp_path = os.path.join(self.exp_dir, self.exp_name)
            # self.img_dir = '/home/byh/CartoonFace/data/personai_icartoonface_dettrain/icartoonface_dettrain/'
            # self.anno_path = '/home/byh/CartoonFace/data/personai_icartoonface_dettrain/icartoonface_dettrain.csv'
            # self.test_dir = '/home/byh/CartoonFace/data/personai_icartoonface_detval/'

            self.img_dir = '/data/byh/CartoonFace/personai_icartoonface_dettrain/icartoonface_dettrain/'
            self.anno_path = '/data/byh/CartoonFace/personai_icartoonface_dettrain/icartoonface_dettrain.csv'
            self.test_dir = '/data/byh/CartoonFace/personai_icartoonface_detval/'

        elif plat is 'cs':
            self.exp_dir = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/exp/'
            self.exp_path = os.path.join(self.exp_dir, self.exp_name)
            self.img_dir = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/' \
                           'data/personai_icartoonface_dettrain/icartoonface_dettrain/'
            self.anno_path = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/' \
                             'data/personai_icartoonface_dettrain/icartoonface_dettrain.csv'

            self.test_dir = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/' \
                            'data/personai_icartoonface_detval/'

        elif plat is 'my':
            self.exp_dir = 'G:\\BBBLib/CartoonFace/exp/'
            self.exp_path = os.path.join(self.exp_dir, self.exp_name)
            self.img_dir = 'G:\\BBBLib/CartoonFace/data/personai_icartoonface_dettrain/icartoonface_dettrain/'
            self.anno_path = 'G:\\BBBLib/CartoonFace/data/personai_icartoonface_dettrain/icartoonface_dettrain_short.csv'
            self.test_dir = 'G:\\BBBLib/CartoonFace/data/personai_icartoonface_detval/'

        elif plat is 'tc':
            pass



class Recogniton_Parameter(Base_parameter):
    def __init__(self):
        super(Recogniton_Parameter, self).__init__()
        # Model Path
        self.resume = False
        self.exp_name = 'xx.pkl'

        # Dataset
        self.train_path = None
        self.set_path()  # 可选：lab, my, tc # 不填写时自动判断lab 或 my

        # Dataset Setting
        self.no_color_aug = False
        self.not_rand_crop = False
        self.shift = 0.1
        self.scale = 0.4
        self.flip = 0.5

        self.input_h = 256
        self.input_w = 256
        self.down_ratio = 4
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio

        # NetWork Setting
        self.model = 'ResNeSt'

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)

        # Optimizer
        self.optimizer = 'SGD'
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-5

        # Loss
        self.reg_loss = 'l1'

        # Rec_Train
        self.train_num = 4480  #4480
        self.BATCH_SIZE = 8
        self.EPOCH = 200
        self.log_step = 50  # 每隔50step，输出一次训练信息

        # Logging

        self.logger.debug('------------ Basic Setting ------------')
        self.logger.debug('Platform   : {}'.format(self.platfrom))
        self.logger.debug('Device     : {}'.format(self.device))
        self.logger.debug('Image Dir  : {}'.format(self.img_dir))
        self.logger.debug('Anno Path   : {}'.format(self.anno_path))
        self.logger.debug('Image Size : h:{},w:{}'.format(self.input_h, self.input_w))
        self.logger.debug('Exp Path   : {}'.format(self.exp_path))
        self.logger.debug('Batch Size : {}'.format(self.BATCH_SIZE))
        self.logger.debug('---------------------------------------')




    def set_path(self, plat=None):

        if plat is None:
            if platform.system() == 'Windows':
                print(platform.system())
                self.platfrom = 'Windows'
                plat = 'my'
            elif platform.system() == 'Linux':
                print(platform.system())
                self.platfrom = 'Linux'
                plat = 'lab'

        if plat is 'lab':
            self.exp_dir = '/home/byh/CartoonFace/exp/'
            self.exp_path = os.path.join(self.exp_dir, self.exp_name)
            # self.img_dir = '/home/byh/CartoonFace/data/personai_icartoonface_dettrain/icartoonface_dettrain/'
            # self.anno_path = '/home/byh/CartoonFace/data/personai_icartoonface_dettrain/icartoonface_dettrain.csv'
            # self.test_dir = '/home/byh/CartoonFace/data/personai_icartoonface_detval/'

            self.img_dir = '/data/byh/CartoonFace/personai_icartoonface_rectrain/icartoonface_rectrain/'
            self.anno_path = '/data/byh/CartoonFace/personai_icartoonface_rectrain/icartoonface_rectrain_det.txt'
            # self.test_dir = '/data/byh/CartoonFace/personai_icartoonface_detval/'


        elif plat is 'cs':
            self.exp_dir = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/exp/'
            self.exp_path = os.path.join(self.exp_dir, self.exp_name)
            self.img_dir = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/' \
                           'data/personai_icartoonface_dettrain/icartoonface_dettrain/'
            self.anno_path = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/' \
                             'data/personai_icartoonface_dettrain/icartoonface_dettrain.csv'
            self.test_dir = '/SISDC_GPFS/Home_SE/kongj-jnu/bianyh-jnu/bianyh-jnu/BBB/iqiyi/' \
                            'data/personai_icartoonface_detval/'

        elif plat is 'my':
            self.exp_dir = 'G:\\BBBLib/CartoonFace/exp/'
            self.exp_path = os.path.join(self.exp_dir, self.exp_name)
            self.img_dir = 'G:\\BBBLib/CartoonFace/data/personai_icartoonface_dettrain/icartoonface_dettrain/'
            self.anno_path = 'G:\\BBBLib/CartoonFace/data/personai_icartoonface_dettrain/icartoonface_dettrain_short.csv'
            self.test_dir = 'G:\\BBBLib/CartoonFace/data/personai_icartoonface_detval/'


        elif plat is 'tc':
            pass

if __name__ == '__main__':
    #para = Detection_Parameter()
    para = Recogniton_Parameter()








