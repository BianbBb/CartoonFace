
import cv2
import os
import torch
import torch.utils.data as DATA
import sys
sys.path.append("..")
import config as cfg
from detect.dataload import DetDataset
from backbone.Hourglass.large_hourglass import HourglassNet
from detect.trainer import DetTrainer
from detect.network import Network

def main(para):
    torch.manual_seed(para.SEED)
    torch.backends.cudnn.benchmark = True

    logger = para.logger
    #device = para.device

    logger.debug('------ Load Network ------')
    network = Network(para)
    net = network.net
    # # from torchsummary import summary
    # #     # summary(net.cuda(),(3,512,512),batch_size=8)
    # #     # print(net)

    logger.debug('------ Load Dataset ------')
    Train_Data = DetDataset(para, flag='train', train_num=para.train_num)
    train_loader = DATA.DataLoader(dataset=Train_Data, batch_size=para.BATCH_SIZE, shuffle=True, drop_last=True)

    Val_Data = DetDataset(para, flag='validation', train_num=para.train_num)
    val_loader = DATA.DataLoader(dataset=Val_Data, batch_size=para.BATCH_SIZE , shuffle=False, drop_last=True)

    logger.debug('------     Train    ------')
    Trainer = DetTrainer(para, net, train_loader=train_loader, val_loader=val_loader, optimizer='SGD')
    Trainer.run()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    parameter = cfg.Detection_Parameter()
    main(parameter)
