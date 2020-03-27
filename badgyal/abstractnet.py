import torch
import badgyal.model as model
import badgyal.net as proto_net
import badgyal.proto.net_pb2 as pb
import chess
from badgyal.board2planes import board2planes, policy2moves, bulk_board2planes
import pylru
import sys



CACHE=40000
MAX_BATCH = 8
MIN_POLICY=0.2


class AbstractNet:

    def __init__(self, cuda=True):
        self.net = self.load_net()
        self.cuda = cuda
        if self.cuda:
            self.net = self.net.cuda()
        self.net.eval()
        self.cache = pylru.lrucache(CACHE)
        self.prefetch = {}

    def process_boards(self, boards):
        input = bulk_board2planes(boards)
        if self.cuda:
            input = input.cuda()
        with torch.no_grad():
            policies, values = self.net(input)
            return policies, values

    def cache_boards(self, boards, softmax_temp=1.61):
        for b in boards:
            epd = b.epd()
            if not epd in self.cache:
                self.prefetch[epd] = b

        if len(self.prefetch) > MAX_BATCH:
            policies, values = self.process_boards(self.prefetch.values())
            with torch.no_grad():
                for i, b in enumerate(self.prefetch.values()):
                    inp = policies[i].unsqueeze(dim=0)
                    policy = policy2moves(b, inp, softmax_temp=softmax_temp)
                    value = values[i].item()
                    self.cache[b.epd()] = [policy, value]
            self.prefetch = {}

    def eval(self, board, softmax_temp=1.61):
        epd = board.epd()
        if epd in self.cache:
            policy, value = self.cache[epd]
        else:
            # put all the child positions on the board
            boards = [board.copy()]
            policies, values = self.process_boards(boards)


            with torch.no_grad():

                for i, b in enumerate(boards):
                    inp = policies[i].unsqueeze(dim=0)
                    policy = policy2moves(b, inp, softmax_temp=softmax_temp)
                    value = values[i].item()
                    self.cache[b.epd()] = [policy, value]

            policy, value = self.cache[epd]

        # get the best move and prefetch it
        tocache = []

        for m, val in policy.items():
            if val >= MIN_POLICY:
                bd = board.copy()
                bd.push_uci(m)
                tocache.append(bd)
        if (len(tocache) < 1):
            m = max(policy, key = lambda k: policy[k])
            bd = board.copy()
            bd.push_uci(m)
            tocache.append(bd)
        self.cache_boards(tocache, softmax_temp=softmax_temp)

        # return the values
        return policy, value
