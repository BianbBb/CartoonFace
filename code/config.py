import numpy as np

_Seed = 2020

_Data = '/data'

_Resume = True
_Det_model = '/exp/det_01.pkl'
_Rec_model = '/exp/rec_01.pkl'

# Detection


class Detection_Parameter():
    def __init__(self):
        self.img_dir = 'G:\BBBLib\CartoonFace\data\personai_icartoonface_dettrain\icartoonface_dettrain\\'
        self.anno_path = 'G:\BBBLib\CartoonFace\data/personai_icartoonface_dettrain/icartoonface_dettrain_short.csv'
        self.max_objs = 5  # 一张image中最多目标数量

        self.no_color_aug = False
        self.not_rand_crop = False
        self.shift = 0.1
        self.scale = 0.4
        self.flip = 0.5

        self.input_h = 512
        self.input_w = 512
        self.down_ratio = 4

        self.reg_offset = True

        self.num_classes = 1

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571], dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)





