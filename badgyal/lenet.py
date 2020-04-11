import torch
import badgyal.model as model
import badgyal.net as proto_net
import badgyal.proto.net_pb2 as pb
import chess
from badgyal.board2planes import board2planes, policy2moves, bulk_board2planes
import pylru
import sys
import os.path
from badgyal import AbstractNet
import math

CHANNELS=128
BLOCKS=10
SE=4

class LENet(AbstractNet):

    def __init__(self, cuda=True):
        super().__init__(cuda=cuda)

    def load_net(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        file = os.path.join(my_path, "LE.pb.gz")
        net = model.Net(CHANNELS, BLOCKS, CHANNELS, SE)
        net.import_proto_classical(file)
        return net

    def value_to_scalar(self, value):
        wdl0 = value[0].item()
        wdl1 = value[1].item()
        wdl2 = value[2].item()
        minv = min(wdl0,wdl1)
        minv = min(minv,wdl2)
        w_ = math.exp(wdl0 - minv)
        d_ = math.exp(wdl1 - minv)
        l_ = math.exp(wdl2 - minv)
        p = (w_ * 1.0 + d_ * 0.5 ) / (w_ + d_ + l_)
        return 2.0*p-1.0;
