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


class BGTorchNet(AbstractNet):

    def __init__(self, cuda=True):
        super().__init__(cuda=cuda)

    def load_net(self):
        my_path = os.path.abspath(os.path.dirname(__file__))
        file = os.path.join(my_path, "badgyal.pt")
        return torch.jit.load(file)
