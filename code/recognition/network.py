import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F

from code.backbone.ResNeSt.resnest import resnest50


class ResNeSt(nn.Module):
    def __init__(self, resume=True):
        super(ResNeSt, self).__init__()
        resnest = resnest50()
        if resume is False:  # 第一次训练时需要加载.pth
            resnest.load_state_dict(torch.load('./exp/resnest50-528c19ca.pth'))

        self.features = resnest


    def forward(self,x):
        out = self.features(x)
        return out


if __name__ == '__main__':

    from code.utils.freeze_layer import freeze

    resnet = ResNeSt()

    Freeze_List = ['features.conv1', 'features.layer2']
    UnFreeze_List = ['features.layer4.2', 'features.fc']
    freeze(resnet, Freeze_List, UnFreeze_List)


    # data_input = torch.Tensor(torch.randn([4, 3, 96, 96]))  # 这里假设输入图片是96x96
    # print(    data_input.size())
    #
    # out = resnet(data_input)
    # print(out.size())
    # from torchsummary import summary
    # summary(resnet, (3, 224, 224),batch_size=1)

