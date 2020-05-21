import sys
sys.path.append("..")
import config as cfg
from backbone.Hourglass.large_hourglass import HourglassNet


class Network():
    def __init__(self, para):
        self.para = para
        self.net = None
        if para.model == 'Hourglass':
            self.base_hourglass()

    def base_hourglass(self):
        self.net = HourglassNet(
            self.para.heads,
            num_stacks=1,
            num_branchs=4,
            #num_branchs=self.para.num_branchs,
            dims=[256, 256, 256, 512, 1024],
            #dims=self.para.dims
        )
        # weights path

