from badgyal.abstractnet import LoadedNet, MultiNet

BGNet = lambda cuda: LoadedNet("badgyal-8.pb.gz", 128, 10, 4, cuda=cuda)
MGNet = lambda cuda: LoadedNet("meangirl-8.pb.gz", 32, 4, 2, cuda=cuda)
LENet = lambda cuda: LoadedNet("LE.pb.gz", 128, 10, 4, classical=False, cuda=cuda)
T59 = lambda cuda: LoadedNet("../../nets/591226.pb.gz", 128, 10, 4, classical=False, cuda=cuda)
T70 = lambda cuda: LoadedNet("../../nets/701494.pb.gz", 128, 10, 4, classical=False, cuda=cuda)
M1 = lambda cuda: MultiNet([BGNet])
