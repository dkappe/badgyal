import torch
import badgyal.model as model
import badgyal.net as proto_net
import badgyal.proto.net_pb2 as pb
import chess
from badgyal.board2planes import board2planes, policy2moves, bulk_board2planes
import pylru
import sys
import os.path
from badgyal import WDLNet
import math

CHANNELS=192
BLOCKS=16
SE=6

class MENet(WDLNet):

    def __init__(self, cuda=True, torchScript=False):
        super().__init__(cuda=cuda, torchScript=torchScript)

    def load_net(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        file = os.path.join(my_path, "ME.pb.gz")
        net = model.Net(CHANNELS, BLOCKS, CHANNELS, SE)
        net.import_proto_classical(file)
        return net
