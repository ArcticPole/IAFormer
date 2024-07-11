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
            # self.data=np.load('../Dataset/Mocap/train_3_120_mocap.npy', allow_pickle=True)


            # print(self.data.shape)
            # use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]
            # self.data=self.data.reshape(self.data.shape[0],3,-1,31,3)
            # self.data=self.data[:,:,:,use,:]
            # self.data=self.data.reshape(self.data.shape[0],3,-1,45)

            self.len=len(self.data)
            print(self.data.shape)
        else:
            if opt.dataset=='Mocap':
                self.data=np.load('../Dataset/Mocap/test_3_75_mocap_umpm.npy')
                # self.data=np.load('../Dataset/Mocap/mix1_6persons.npy')
                # self.data=np.load('../Dataset/Mocap/mix2_10persons.npy')
                # self.data=np.load('../Dataset/Mocap/test_3_120_mocap.npy', allow_pickle=True)

            if opt.dataset=='MuPoTS':
                self.data=np.load('../Dataset/mupots-3d-eval/mupots_120_3persons.npy')

            print(self.data.shape)
            self.len=len(self.data)

        if self.opt.seq_len == 35 and self.process_mode == 'load':
            if os.path.exists('../Dataset/Mocap/train_3_35_sr_{}.npy'.format(self.sample_rate)) and mode == 'train':
                self.new_data = np.load('../Dataset/Mocap/train_3_35_sr_{}.npy'.format(self.sample_rate))
            elif os.path.exists('../Dataset/Mocap/test_3_35_sr_{}.npy'.format(self.sample_rate)):
                self.new_data = np.load('../Dataset/Mocap/test_3_35_sr_{}.npy'.format(self.sample_rate))
            self.len=len(self.new_data)
            print(self.new_data.shape)

        if self.opt.seq_len == 35 and self.process_mode == 'init':
            self.new_data = None
            num_sample = self.data.shape[0]
            for sam_idx in range(num_sample):
                if sam_idx % 100 == 0:
                    print(sam_idx)
                data_process = self.data[sam_idx]
                # print(data_process.shape)
                # exit(0)
                num_frames = data_process.shape[1]

                # print(np.arange(0, num_frames, self.sample_rate))
                data = data_process#[:, np.arange(0, num_frames, self.sample_rate), :]
                num_frames = data.shape[1]
                # print(num_frames)
                # print(data.shape)
                fs = np.arange(0, num_frames - self.opt.seq_len + 1, self.sample_rate)  # seq_len: past frame length + future frame length
                fs_sel = fs  # fs是可以作为起始帧的索引，fs_sel 可以被认为是在完整动作序列中提取子序列的索引集合。
                # print(fs)
                for k in np.arange(0, self.opt.seq_len - 1):  # for sample in rate
                    fs_sel = np.vstack((fs_sel, fs + k + 1))  # 这样一来fs_sel中每一列都代表一个完整的seq_len
                # print(fs_sel)
                fs_sel = fs_sel.transpose()  # 经过转置之后每一行都代表一个完整的seq_len
                # print(np.shape(data))

                seq_sel = data[:, fs_sel, :]
                seq_sel = seq_sel.transpose(1, 0, 2, 3)
                # print(seq_sel.shape)
                # exit(0)
                if self.new_data is None:
                    self.new_data = seq_sel
                else:
                    self.new_data = np.concatenate((self.new_data, seq_sel), axis=0)
                    # mod_data.append(seq_sel)
                    # mod_data = np.array(mod_data)
            self.len=len(self.new_data)
            print(self.new_data.shape)
            if mode == 'train':
                np.save('../Dataset/Mocap/train_3_35_sr_{}.npy'.format(self.sample_rate), self.new_data)
            else:
                np.save('../Dataset/Mocap/test_3_35_sr_{}.npy'.format(self.sample_rate), self.new_data)

    def __getitem__(self, index):
        # index = 700
        # print(index)
        if self.opt.dataset == 'Mocap' and self.opt.seq_len == 75:
            # input_seq=self.data[index][:,:2*self.opt.frame_in,:][:,::2,:]  # 4second 120frame video resample to 60 frame
            # output_seq=self.data[index][:,:2*self.opt.seq_len,:][:,::2,:]  # first 30 frame use as input, last 30 frame use as output

            input_seq=self.data[index][:, :self.opt.frame_in, :]  # 4second 120frame video resample to 60 frame
            output_seq=self.data[index][:, :self.opt.seq_len, :]  # first 30 frame use as input, last 30 frame use as output
        elif self.opt.dataset == 'Mocap' and self.opt.seq_len == 35:
            # print('aaaaaa')
            input_seq=self.new_data[index][:, :self.opt.frame_in, :]  # 4second 120frame video resample to 60 frame
            output_seq=self.new_data[index][:, :self.opt.seq_len, :]  # first 30 frame use as input, last 30 frame use as output
        else:
            input_seq=self.data[index][:,:2*self.opt.frame_in,:][:,::2,:]  # 4second 120frame video resample to 60 frame
            output_seq=self.data[index][:,:,:][:,::2,:]  # first 30 frame use as input, last 30 frame use as output
        # last_input=input_seq[:,-1:,:]
        # output_seq=np.concatenate([last_input,output_seq],axis=1)

        # print(input_seq.shape)
        pad_idx = np.repeat([self.opt.frame_in - 1], self.opt.frame_out)
        i_idx = np.append(np.arange(0, self.opt.frame_in), pad_idx)
        # print(i_idx)
        # input_seq = input_seq.transpose(0, 2, 1)
        # input_seq = input_seq[:, i_idx, :]
        # output_seq = output_seq.transpose(0, 2, 1)

        input_seq = input_seq.transpose(0, 2, 1)
        input_seq = input_seq[:, :, i_idx]
        output_seq = output_seq.transpose(0, 2, 1)

        # output_seq = output_seq[:, :, self.opt.frame_in:]
        # print(input_seq.shape, output_seq.shape)
        # print(input_seq[0][0]==output_seq[0][0])
        # print(input_seq[0][0])

        # exit(0)
        return input_seq, output_seq

    def __len__(self):
        return self.len



class TESTDATA(data.Dataset):
    def __init__(self,dataset='mocap'):

        if dataset=='mocap':
            self.data=np.load('./mocap/test_3_120_mocap.npy',allow_pickle=True)


            # use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]
            # self.data=self.data
            # self.data=self.data.reshape(self.data.shape[0],self.data.shape[1],-1,31,3)
            # self.data=self.data[:,:,:,use,:]
            # self.data=self.data.reshape(self.data.shape[0],self.data.shape[1],-1,45)

        if dataset=='mupots':
            self.data=np.load('./mupots3d/mupots_120_3persons.npy',allow_pickle=True)

        self.len=len(self.data)

    def __getitem__(self, index):

        input_seq=self.data[index][:,:30,:][:,::2,:]#input, 30 fps to 15 fps
        output_seq=self.data[index][:,30:,:][:,::2,:]#output, 30 fps to 15 fps
        last_input=input_seq[:,-1:,:]
        output_seq=np.concatenate([last_input,output_seq],axis=1)

        return input_seq,output_seq

    def __len__(self):
        return self.len
