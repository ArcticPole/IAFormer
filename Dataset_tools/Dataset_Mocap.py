import torch.utils.data as data
import torch
import numpy as np
import os

class Datasets(data.Dataset):
    def __init__(self, opt, mode):
        self.opt = opt
        # self.process_mode = 'init'
        self.process_mode = 'load'
        self.sample_rate = 15
        if mode == 'train':
            self.data=np.load('../Dataset/Mocap/train_3_75_mocap_umpm.npy')

            self.len=len(self.data)
            print(self.data.shape)
        else:

            self.data=np.load('../Dataset/Mocap/test_3_75_mocap_umpm.npy')
            # self.data=np.load('../Dataset/Crowd/mix1_6persons.npy')
            # self.data=np.load('../Dataset/Crowd/mix2_10persons.npy')

            print(self.data.shape)
            self.len=len(self.data)

        if self.opt.seq_len == 35 and self.process_mode == 'load':
            if os.path.exists('../Dataset/Mocap/train_3_35_sr_{}.npy'.format(self.sample_rate)) and mode == 'train':
                self.new_data = np.load('../Dataset/Mocap/train_3_35_sr_{}.npy'.format(self.sample_rate))
            elif os.path.exists('../Dataset/Mocap/test_3_35_sr_{}.npy'.format(self.sample_rate)):
                self.new_data = np.load('../Dataset/Mocap/test_3_35_sr_{}.npy'.format(self.sample_rate))
            self.len=len(self.new_data)
            print(self.new_data.shape)


    def __getitem__(self, index):

        if self.opt.dataset == 'Mocap' and self.opt.seq_len == 75:
            input_seq=self.data[index][:, :self.opt.frame_in, :]  
            output_seq=self.data[index][:, :self.opt.seq_len, :]  

        elif self.opt.dataset == 'Mocap' and self.opt.seq_len == 35:
            input_seq=self.new_data[index][:, :self.opt.frame_in, :]
            output_seq=self.new_data[index][:, :self.opt.seq_len, :]
        else:
            input_seq=self.data[index][:,:2*self.opt.frame_in,:][:,::2,:]
            output_seq=self.data[index][:,:,:][:,::2,:]


        # print(input_seq.shape)
        pad_idx = np.repeat([self.opt.frame_in - 1], self.opt.frame_out)
        i_idx = np.append(np.arange(0, self.opt.frame_in), pad_idx)


        input_seq = input_seq.transpose(0, 2, 1)
        input_seq = input_seq[:, :, i_idx]
        output_seq = output_seq.transpose(0, 2, 1)

        return input_seq, output_seq

    def __len__(self):
        return self.len



class TESTDATA(data.Dataset):
    def __init__(self,dataset='mocap'):

        if dataset=='mocap':
            self.data=np.load('./mocap/test_3_120_mocap.npy',allow_pickle=True)


        self.len=len(self.data)

    def __getitem__(self, index):

        input_seq=self.data[index][:,:30,:][:,::2,:]
        output_seq=self.data[index][:,30:,:][:,::2,:]
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)

        return input_seq,output_seq

    def __len__(self):
        return self.len
