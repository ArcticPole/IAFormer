import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from model.layers import TimeCon
from model.layers import IPM as IP
from model.layers import Attention as Att
from model.layers import GCN

from model.layers import IPM_IAW as AP
from model.layers import IPM_ITW as LP
from model.layers import IPLM as CP

from model.layers import PE



class IAFormer(nn.Module):
    def __init__(self, seq_len, d_model, opt, num_kpt, dataset):
        super(IAFormer, self).__init__()
        self.opt = opt

        self.mid_feature = opt.seq_len
        self.dataset = dataset
        self.seq_len = seq_len
        self.num_kpt = num_kpt
        self.dct, self.idct = self.get_dct_matrix(self.num_kpt*3)

        if self.opt.exp == 'Mocap':
            self.w_sp = self.opt.w_sp
            self.w_tp = self.opt.w_tp
            self.w_cb = self.opt.w_cb
        else:
            self.w_sp = 1
            self.w_tp = 1
            self.w_cb = 1

        self.Att = Att.TransformerDecoder(num_layers=self.opt.num_layers,
                                          num_heads=5,
                                          d_model=self.mid_feature,
                                          d_ff=self.mid_feature,
                                          dropout=0.1)  # 6
    

        self.GCNQ1 = GCN.GCN(input_feature=self.mid_feature,
                             hidden_feature=d_model,
                             p_dropout=0.3,
                             num_stage=self.opt.num_stage,
                             node_n=num_kpt * 3)#2

        self.GCNQ2 = GCN.GCN(input_feature=self.mid_feature,
                             hidden_feature=d_model,
                             p_dropout=0.3,
                             num_stage=self.opt.num_stage,
                             node_n=num_kpt * 3)


        self.IP = IP.IP(opt=self.opt, dim_in=self.mid_feature,
                        mid_feature=self.mid_feature, num_axis=num_kpt * 3, dropout=0.1)

        self.timecon = TimeCon.timecon_plus()

        self.AP = AP.AP(self.opt, in_features=self.opt.frame_in,
                        hidden_features=self.mid_feature, out_features = self.mid_feature)

        self.CP = CP.CP(self.opt)

        self.PE = PE.PositionalEmbedding(opt=self.opt, mid_feature=self.mid_feature, embed_size=opt.batch_size)

        

    def forward(self, input_ori, gt):

        input = torch.matmul(self.dct, input_ori)

        if self.dataset == "Mocap" or self.dataset == "CHI3D":
            input = input
        elif self.dataset == "Human3.6M":
            input = input.unsqueeze(dim=1)
            input_ori = input_ori.unsqueeze(dim=1)
            gt = gt.unsqueeze(dim=1)

        num_person = np.shape(input)[1]


        for i in range(num_person):
            people_in = input[:, i, :, :].clone().detach()
            if i == 0:
                people_feature_all = self.GCNQ1(people_in).unsqueeze(1).clone()
            else:
                people_feature_all = torch.cat([people_feature_all, self.GCNQ1(people_in).unsqueeze(1).clone()], 1)


        for bat_idx in range(self.opt.batch_size):
            itw = LP.Trajectory_Weight(self.opt, input_ori[bat_idx])
            if bat_idx == 0:
                bat_itw = itw.unsqueeze(0)
            else:
                bat_itw = torch.cat([bat_itw, itw.unsqueeze(0)], 0)


        IP_score = self.IP(people_feature_all.clone(), input_ori.clone(), num_person, bat_itw, self.AP)

        CP_score, k_loss, CP_ema = self.CP(IP_score.clone())
        IP_feature = IP_score
        CP_ema = CP_ema * CP_score


        for i in range(num_person):

            people_feature = people_feature_all[:, i, :, :]
            
            filter_feature = people_feature.clone().detach()

            Pe = self.PE.forward(idx=i)

            feature_att = self.Att(filter_feature, IP_feature, memo2=CP_ema, embedding=Pe)
            feature_att += people_feature.clone()

            feature = self.GCNQ2(feature_att)
            feature = torch.matmul(self.idct, feature)
            feature = feature.transpose(1, 2)

            if i == 0:
                predic = feature.unsqueeze(1).clone()
            else:
                predic = torch.cat([predic, feature.unsqueeze(1).clone()], 1)


        loss = self.mix_loss(predic, gt) + self.w_cb * k_loss


        if self.dataset == "Mocap" or self.dataset == "CHI3D":
            return predic, loss
        elif self.dataset == "Human3.6M":
            return predic[:, 0, :, :], loss

    def mix_loss(self, predic, gt):

        gt = gt.transpose(2, 3)
        bs, np, sql, _ = gt.shape

        spacial_loss_pred = torch.mean(torch.norm((predic[:, :, self.opt.frame_in:, :] - gt[:, :, self.opt.frame_in:, :]), dim=3))
        spacial_loss_ori = torch.mean(torch.norm((predic[:, :, :self.opt.frame_in, :] - gt[:, :, :self.opt.frame_in, :]), dim=3))
        spacial_loss = spacial_loss_pred + spacial_loss_ori * 0.1

        temporal_loss = 0


        for idx_person in range(np):

            
            tempo_pre = self.timecon(predic[:, idx_person, :, :].unsqueeze(1))
            tempo_ref = self.timecon(gt[:, idx_person, :, :].unsqueeze(1))
            
            temporal_loss += torch.mean(torch.norm(tempo_pre-tempo_ref, dim=3))

        loss = self.w_sp * spacial_loss + self.w_tp * temporal_loss

        return loss


    def get_dct_matrix(self, N):
        # Computes the discrete cosine transform (DCT) matrix and its inverse (IDCT)
        dct_m = np.eye(N)
        for k in np.arange(N):
            for i in np.arange(N):
                w = np.sqrt(2 / N)
                if k == 0:
                    w = np.sqrt(1 / N)
                dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
        idct_m = np.linalg.inv(dct_m)
        dct_m = torch.tensor(dct_m).float().cuda()
        idct_m = torch.tensor(idct_m).float().cuda()
        return dct_m, idct_m



class mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x