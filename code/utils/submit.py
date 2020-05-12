import os
import time
import numpy as np

def extract_feature(imgpath,box):
    feat = []
    return feat

def gen_bins(personai_icartoonface_rectest_path,output_bin_path):
    feats=[]
    with open(os.path.join(personai_icartoonface_rectest_path,'icartoonface_rectest_det.txt'),\
              'r',encoding='utf-8') as f:
        for line in f.readlines():
            s_time=time.time()
            line_info = line.strip().split()
            imgpath = os.path.join(personai_icartoonface_rectest_path,'icartoonface_rectest',\
                                   line_info[0])
            box = [int(v) for v in line_info[1:5]]
            feat = extract_feature(imgpath,box)
            feats.append(list(feat))
            print('{}/22500,time cost: {}s'.format(len(feats),time.time()-s_time))
    np.asarray(feats, dtype=float).tofile(output_bin_path)

if __name__=='__main__':
    personai_icartooonface_rectest_path = 'personai_icartoonface_rectest'
    output_bin_path = 'output.bin'
    gen_bins(personai_icartooonface_rectest_path,output_bin_path)