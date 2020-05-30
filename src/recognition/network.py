import torch
import torch.nn as nn
import sys
sys.path.append("..")
from backbone.ResNeSt.resnest import resnest50
from backbone.ResNeSt.resnet import ResNet, Bottleneck
from backbone.ftNet.model import ft_net
## TODO: 按照名称搭建不同网络
## ResNeSt最后加一层Sigmoid进行约束


class Network():
    def __init__(self, para):
        self.para = para
        self.net = None

        if para.model == 'ResNeSt':
            self.base_resnest()
        elif para.model == 'ResNest_sigmoid':
            self.resnest_sigmoid()
        elif para.model == 'ResNest_tanh_fc':
            self.resnest_tanh_fc()
        elif para.model == 'txf':


    def base_resnest(self):
        self.net = ResNeSt()

    def resnest_sigmoid(self):
        self.net = ResNeSt_Sigmoid()

    def resnest_tanh_fc(self):
        self.net = ResNeSt_Tanh()




class ResNeSt_Sigmoid(nn.Module):
    def __init__(self, para=None):
        super(ResNeSt_Sigmoid, self).__init__()
        resnest = resnest50()
        self.features = resnest
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        out = self.features(x)
        out = self.sigmoid(out)
        return out


class ResNeSt_Tanh(nn.Module):
    def __init__(self, para=None):
        super(ResNeSt_Tanh, self).__init__()


        resnest = self.resnest101()

        self.features = resnest
        self.fc = nn.Linear(1024, 512)
        self.bn = nn.BatchNorm1d(512)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.features(x)
        out = self.fc(out)
        out = self.bn(out)
        out = self.tanh(out)
        return out



    def resnest101(pretrained=False, root='~/.encoding/models', **kwargs):
        model = ResNet(Bottleneck, [3, 4, 23, 3],num_classes=1024,
                       radix=2, groups=1, bottleneck_width=64,
                       deep_stem=True, stem_width=64, avg_down=True,
                       avd=True, avd_first=False, **kwargs)
        return model



class ResNeSt(nn.Module):
    def __init__(self, para=None):
        super(ResNeSt, self).__init__()
        resnest = resnest50()
        # if resume is False:  # 第一次训练时需要加载.pth 比赛中禁用预训练模型
        #     resnest.load_state_dict(torch.load('./exp/resnest50-528c19ca.pth'))
        self.features = resnest


    def forward(self,x):
        out = self.features(x)
        return out




if __name__ == '__main__':
    from utils.freeze_layer import freeze
    import config as cfg
    parameter = cfg.Detection_Parameter()

    resnet = ResNeSt_Tanh(parameter)
    # Freeze_List = ['features.conv1', 'features.layer2']
    # UnFreeze_List = ['features.layer4.2', 'features.fc']
    # freeze(resnet, Freeze_List, UnFreeze_List)

    print(resnet)


    # data_input = torch.Tensor(torch.randn([4, 3, 96, 96]))  # 这里假设输入图片是96x96
    # print(    data_input.size())
    #
    # out = resnet(data_input)
    # print(out.size())
    from torchsummary import summary
    summary(resnet, (3,256,256), batch_size=2)

