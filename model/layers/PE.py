import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, opt, mid_feature, embed_size, dropout=0.1):
        super().__init__()
        # self.N = num_person
        self.J = opt.in_features//3
        self.feature = opt.in_features
        self.dim = 3
        self.time = mid_feature
        self.joint = nn.Parameter(torch.zeros(self.J, embed_size))
        self.person = nn.Parameter(torch.zeros(1, embed_size))
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, idx):
        torch.nn.init.normal_(self.joint, std=.02)
        torch.nn.init.normal_(self.person, std=.02, mean=idx)
        p_person = self.person.repeat_interleave(self.feature, dim=0)
        p_joint = self.joint.repeat(self.dim, 1)
        # print(p_person.shape, p_joint.shape)
        p = p_person + p_joint
        p = p.unsqueeze(0)
        # print(p.shape)
        p = p.transpose(2, 0)
        # print(p.shape)
        p = p.repeat(1, 1, self.time)
        # print(p.shape)
        return self.dropout(p)