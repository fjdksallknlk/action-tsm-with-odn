import torch
from torch import nn

class CPL(torch.nn.Module):
    def __init__(self, n_classes, in_feat_dim, out_feat_dim, consensus,
                is_shift=False, temporal_pool=False, num_segments=4, init_weight=True, K=1):

        super(CPL, self).__init__()
        self.n_classes = n_classes
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.consensus = consensus
        self.is_shift = is_shift
        self.temporal_pool = temporal_pool
        self.num_segments = num_segments
        # self.conv = nn.Conv2d(in_feat_dim, out_feat_dim, 1, 1, 0, bias=False)
        # self.bn = nn.BatchNorm2d(out_feat_dim)
        # self.relu = nn.ReLU6(inplace=True)
        self.linear = nn.Linear(in_feat_dim, out_feat_dim)
        # K centers per class, default 1, centers => out_feat_dim * n_classes
        if K == 1:
            self.centers = nn.Parameter(torch.randn(self.out_feat_dim, self.n_classes).cuda(), requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):

        # x [batch_size*num_seqments, out_feat_dim]
        x = self.linear(x)

        # x [batch_size, num_segments, 2048]
        if self.is_shift and self.temporal_pool:
            x = x.view((-1, self.num_segments // 2) + x.size()[1:])
        else:
            x = x.view((-1, self.num_segments) + x.size()[1:])
        # x [batch_size, 1, num_classes]
        x = self.consensus(x)
        # x [batch_size, in_feat_dim]
        x = x.squeeze(1)

        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2*torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return x, self.centers, -dist