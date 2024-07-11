# amplitude perception
import torch
import torch.nn as nn
import numpy as np
from model.layers import Attention as Att

class AP(nn.Module):
    """
    Amplitude_Weight
    """
    def __init__(self, opt, in_features, hidden_features, out_features):
        super(AP, self).__init__()
        self.opt = opt
        if opt.dataset == '3dpw':
            self.AP_att = Att.TransformerEncoder(hidden_dim=in_features, num_layers=5, num_heads=4, dropout=0.1)
        else:
            self.AP_att = Att.TransformerEncoder(hidden_dim=in_features, num_layers=5, num_heads=5, dropout=0.1)
        self.AP_FC = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.ReLU(),
            # nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Dropout(0.1)
        )

    def forward(self, x):

        motion_oldall = torch.cat([x[:, :, 0].unsqueeze(2), x[:, :, :self.opt.frame_in-1]], dim=2)

        Change = x[:, :, :self.opt.frame_in] - motion_oldall

        feature_am, feature_score = self.AP_att(Change)

        AP_score = self.AP_FC(feature_am)

        return AP_score, Change
