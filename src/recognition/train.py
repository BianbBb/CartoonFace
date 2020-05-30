
import cv2
import os
import torch
import torch.utils.data as DATA
import sys
sys.path.append("..")
import config as cfg
from recognition.dataload import RecDataset

from recognition.trainer import RecTrainer
from recognition.network import Network



def main(para):
    torch.manual_seed(para.SEED)
    torch.backends.cudnn.benchmark = True

    logger = para.logger
    #device = para.device

    logger.debug('------ Load Network ------')
    network = Network(para)
    net = network.net

    # from torchsummary import summary
    # summary(net, (3,512,512),batch_size=8)
    # print(net)

    logger.debug('------ Load Dataset ------')

    Train_Data = RecDataset(para, flag='train', train_num=para.train_num)
    train_loader = DATA.DataLoader(dataset=Train_Data, batch_size=para.BATCH_SIZE, shuffle=True, drop_last=True)

    Val_Data = RecDataset(para, flag='validation', train_num=para.train_num)
    val_loader = DATA.DataLoader(dataset=Val_Data, batch_size=para.BATCH_SIZE, shuffle=False, drop_last=True)

    logger.debug('Train Data Number:{}'.format(len(Train_Data)))
    logger.debug('Validation Data Number:{}'.format(len(Val_Data)))
    # ################
    # def Torch_to_CV2(image):
    #     image = image.numpy()  # 这里index表示anchor/pos/neg
    #     image = image[0].transpose([1, 2, 0])  # CHW - >HWC #这里index表示batch
    #     image = image[..., ::-1]
    #     return image
    #
    # for step, (images, label) in enumerate(train_loader):
    #     print(step)
    #
    #     anchor = images[0]
    #     pos = images[1]
    #     neg = images[2]
    #
    #     import numpy as np
    #     print(type(images))
    #     print(np.shape(anchor))
    #     print(label)
    #     print(type(label))
    #     input = anchor
    #     input = Torch_to_CV2(input)
    #
    #     cv2.imshow('inp',input)
    #     cv2.waitKey(0)

    # #################

    logger.debug('------     Train    ------')
    Trainer = RecTrainer(para, net, train_loader=train_loader, val_loader=val_loader, optimizer='SGD')
    Trainer.run()


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    parameter = cfg.Recogniton_Parameter()
    main(parameter)
