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
        min_val = min(wdl0, wdl1, wdl2)
        w_val = math.exp(wdl0 - min_val)
        d_val = math.exp(wdl1 - min_val)
        l_val = math.exp(wdl2 - min_val)
        p = (w_val * 1.0 + d_val * 0.5 ) / (w_val + d_val + l_val)
        return 2.0*p-1.0;
