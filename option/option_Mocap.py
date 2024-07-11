import os
import argparse
from pprint import pprint
from utils import other_utils as ou

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        """
        :return: option of project
        """

        "---basic option---"
        self.parser.add_argument('--dataset', type=str, default='Mocap', help='used dataset name')
        self.parser.add_argument('--ckpt', type=str, default='../model/checkpoint', help='path of checkpoint')
        self.parser.add_argument('--tensorboard', type=str, default='../model/tensorboard/', help='path to save tensorboard log')
        self.parser.add_argument('--model', type=str, default='IAFormer', help='model type used')
        self.parser.add_argument('--cudaid', type=int, default=7, help='cuda index used')

        "---codebook option---"
        self.parser.add_argument('--codebook_size', type=int, default=256, help='size of codebook(IKS)')
        self.parser.add_argument('--latent_dim', type=int, default=75, help='vector dim in codebook(IKS)')
        self.parser.add_argument('--beta', type=int, default=0.5, help='hyperpara in codebook(IKS), degree of iteration')

        "---hyperpara option---"
        self.parser.add_argument('--drop_out', type=float, default=0.1, help='drop out probability')
        self.parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
        self.parser.add_argument('--epoch', type=int, default=80)
        self.parser.add_argument('--batch_size', type=int, default=96)
        self.parser.add_argument('--lr_now', type=float, default=0.01)
        self.parser.add_argument('--lr_decay_rate', type=float, default=0.98)
        self.parser.add_argument('--max_norm', type=float, default=10000)
        self.parser.add_argument('--in_features', type=int, default=45, help='dim of input feature, n x j')
        self.parser.add_argument('--frame_in', type=int, default=50,
                                 help='input frame number used in dataloader')
        self.parser.add_argument('--frame_out', type=int, default=25,
                                 help='output frame number used in dataloader')
        self.parser.add_argument('--seq_len', type=int, default=75,
                                 help='frame number each sample')

        "---module option---"
        self.parser.add_argument('--use_IP', type=bool, default=True)
        # self.parser.add_argument('--use_IP', type=bool, default=False)
        self.parser.add_argument('--use_AP', type=bool, default=True)
        # self.parser.add_argument('--use_AP', type=bool, default=False)
        self.parser.add_argument('--use_LP', type=bool, default=True)
        # self.parser.add_argument('--use_LP', type=bool, default=False)
        self.parser.add_argument('--use_CP', type=bool, default=True)
        # self.parser.add_argument('--use_CP', type=bool, default=False)
        self.parser.add_argument('--use_TC', type=bool, default=True)
        # self.parser.add_argument('--use_TC', type=bool, default=False)
        


        "---size option of module---"
        self.parser.add_argument('--num_stage', type=int, default=5, help='for GCN')
        self.parser.add_argument('--num_layers', type=int, default=5, help='for self-attention')


        "---execute option---"
        # self.parser.add_argument('--mode', type=str, default='train', help='mode of execute')
        self.parser.add_argument('--mode', type=str, default='test', help='mode of execute')
        self.parser.add_argument('--test_epoch', type=int, default=None,
                                 help='check the model with corresponding epoch')
        self.parser.add_argument('--save_results', type=bool, default=1,
                                 help='whether to save result')
        

        "---Parameter sensitivity experiment---"
        self.parser.add_argument('--w_sp', type=float, default=1)
        self.parser.add_argument('--w_tp', type=float, default=1)
        self.parser.add_argument('--w_cb', type=float, default=1)



    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self, makedir=True):
        self._initial()
        self.opt = self.parser.parse_args()

        if self.opt.model == 'STAGE_4':
            self.opt.d_model = 16
        
        log_name = 'exp_{}_{}_in{}_out{}_IP&AP&LP&CP_{}&{}&{}&{}_lr_{}_lrd_{}_bs_{}_ep_{}_{}{}_cb_{}'.format(self.opt.dataset,
                                                                                           self.opt.model,
                                                                                           self.opt.frame_in,
                                                                                           self.opt.frame_out,
                                                                                           self.opt.use_IP,
                                                                                           self.opt.use_AP,
                                                                                           self.opt.use_LP,
                                                                                           self.opt.use_CP,
                                                                                           self.opt.lr_now,
                                                                                           self.opt.lr_decay_rate,
                                                                                           self.opt.batch_size,
                                                                                           self.opt.epoch,
                                                                                           self.opt.num_stage,
                                                                                           self.opt.num_layers,
                                                                                           self.opt.codebook_size

                                                                                           )
        if log_name == 'exp_Mocap_IAFormer_in50_out25_IP&AP&LP&CP_True&True&True&True_lr_0.01_lrd_0.98_bs_96_ep_80_55_cb_256':
            self.opt.mode = 'test'
        self.opt.exp = log_name

        ckpt = os.path.join(self.opt.ckpt, self.opt.exp)
        if makedir==True:
            if not os.path.isdir(ckpt):
                os.makedirs(ckpt)
                ou.save_options(self.opt, dataset="HM")
            self.opt.ckpt = ckpt
            ou.save_options(self.opt, dataset="HM")

        self._print()

        return self.opt