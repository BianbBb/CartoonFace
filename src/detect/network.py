import sys
sys.path.append("..")
import config as cfg
from backbone.Hourglass.large_hourglass import HourglassNet
from backbone.Hourglass.hourglass_ReLU_BN import HourglassNet_RB
from backbone.Hourglass.hourglass_SK import Hourglass_SK

class Network():
    def __init__(self, para):
        self.para = para
        self.net = None
        if para.model == 'Hourglass':
            self.base_hourglass()
        elif para.model == 'Hourglass_3':
            self.hourglass_rb(self.para.heads, num_stacks=1, num_branchs=4, dims=[256, 256, 256, 256],)
        elif para.model == 'Hourglass_SK':
            self.hourglass_sk()

    def hourglass_sk(self,):
        self.net = Hourglass_SK(n=4, input_c=3, WH=512)

    def base_hourglass(self, heads=None, num_stacks=1, num_branchs=4, dims=[256, 256, 256, 512, 1024],):
        self.net = HourglassNet(
            heads=self.para.heads,
            num_stacks=num_stacks,
            num_branchs=num_branchs,
            dims=dims,
        )
        # weights path
    def hourglass_rb(self, heads=None, num_stacks=1, num_branchs=4, dims=[256, 256, 256, 512, 1024],):
        self.net = HourglassNet_RB(
            heads=self.para.heads,
            num_stacks=num_stacks,
            num_branchs=num_branchs,
            dims=dims,
        )

if __name__ == '__main__':
    print('start')
