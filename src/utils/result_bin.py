
import os
import time
import numpy as np
import sys
sys.path.append("..")
from recognition.test import Tester
import config as cfg


def extract_feature(img_path, box, tester_):
    feat = tester_.cal_feature(img_path, box)
    return feat

def gen_bins(personai_icartoonface_rectest_path,output_bin_path,tester):
    feats=[]
    with open(os.path.join(personai_icartoonface_rectest_path,'icartoonface_rectest_det.txt'),\
              'r',encoding='utf-8') as f:
        for line in f.readlines():
            s_time=time.time()
            line_info = line.strip().split()
            imgpath = os.path.join(personai_icartoonface_rectest_path,'icartoonface_rectest',\
                                   line_info[0])
            box = [int(v) for v in line_info[1:5]]
            feat = extract_feature(imgpath, box,tester)
            feats.append(list(feat))
            print('{}/22500,time cost: {}s'.format(len(feats),time.time()-s_time))
    np.asarray(feats, dtype=float).tofile(output_bin_path)

if __name__=='__main__':
    personai_icartooonface_rectest_path = '/data/byh/CartoonFace/personai_icartoonface_rectest/'
    output_bin_path = '/home/byh/CartoonFace/result/output.bin'
    parameter = cfg.Recogniton_Parameter()
    tester = Tester(parameter)
    gen_bins(personai_icartooonface_rectest_path,output_bin_path,tester)