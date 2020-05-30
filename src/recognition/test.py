import cv2
import numpy as np
import torch
from recognition.network import Network
from config import Recogniton_Parameter
import os


class Tester():
    def __init__(self,para):
        super(Tester, self).__init__()
        # 加载网络
        self.para = para
        self.logger = para.logger
        network = Network(para)
        self.net = network.net
        self.exp_path = os.path.join(self.para.exp_dir, self.para.exp_name)
        self.net.to(self.para.device)
        self.load_weight()
        self.net.eval()

    def cal_feature(self, img_path, box):
        img_ = cv2.imread(img_path)
        img = self.Clip(img_, box)
        img = cv2.resize(img, (256,256))

        inp = (img - self.para.mean) / self.para.std
        inp = inp.transpose(2, 0, 1)
        
        with torch.no_grad():
            torch.cuda.empty_cache()
            inp = torch.unsqueeze(torch.from_numpy(inp), dim=0).float()
            inp = inp.to(device=self.para.device, non_blocking=True)
            output = self.net(inp)
        feat = output.cpu().numpy()
        return feat

    def load_weight(self):
        try:
            self.net.load_state_dict(torch.load(self.exp_path))
            self.logger.info('Net Parameters Loaded Successfully!')
        except FileNotFoundError:
            self.logger.warning('Can not find feature.pkl !')

    def Clip(self, image, box):
        x_min, y_min, x_max, y_max = box
        return image[y_min:y_max, x_min:x_max, :]