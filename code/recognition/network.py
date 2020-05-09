import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

from code.backbone.ResNeSt.resnest import resnest50


class ResNeSt(nn.Module):
    def __init__(self,resume = True):
        super(ResNeSt, self).__init__()
        resnest = resnest50()
        if resume == False: #第一次训练时需要加载.pth
            resnest.load_state_dict(torch.load('./exp/resnest50-528c19ca.pth'))

        self.features = resnest


    def forward(self,x):
        out = self.features(x)
        return out


if __name__ == '__main__':

    from Utils import Is_Contain

    resnet = ResNeSt()

    Freeze_List = ['features.conv1','features.layer2']
    UnFreeze_List = ['features.layer4.2','features.fc']

    for name, value in resnet.named_parameters():
        if Is_Contain(name,UnFreeze_List):
            value.requires_grad = True
        else:
            value.requires_grad = False
        # if 'features.conv1' in name:
        #      value.requires_grad = False
        # if 'features.21' in name:
        #     value.requires_grad = True
        print('name: {0},\t grad: {1}'.format(name, value.requires_grad))



    # data_input = torch.Tensor(torch.randn([4, 3, 96, 96]))  # 这里假设输入图片是96x96
    # print(    data_input.size())
    #
    # out = resnet(data_input)
    # print(out.size())
    # from torchsummary import summary
    # summary(resnet, (3, 224, 224),batch_size=1)

