import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
sys.path.append('..')
# print(sys.path)
from tensorboardX import SummaryWriter
from utils import other_utils as util
from utils import View_skeleton as view3d
# from IPython import embed
from tqdm import tqdm

from option.option_Mocap import Options
# from Utils import util, data_utils, vis_2p
# from Utils.rigid_align import rigid_align_torch
from Dataset_tools import Dataset_Mocap as datasets
from model import IAFormer as model

from torchstat import stat
from collections import OrderedDict
# import torchvision.models as models
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# from model import xiao_model_codebook

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



def main(opt):
    torch.manual_seed(1234567890)
    torch.manual_seed(1234567890)   
    torch.cuda.manual_seed(1234567890) 
    torch.backends.cudnn.deterministic = True

    select_cuda = opt.cudaid
    torch.cuda.set_device(device=select_cuda)
    print("The using GPU is device {0}".format(select_cuda))
    torch.autograd.set_detect_anomaly(True)
    if opt.mode == 'train':
        print('>>> DATA loading >>>')
        dataset = datasets.Datasets(opt, mode='train')
        eval_dataset = datasets.Datasets(opt, mode='test')

        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        eval_data_loader = DataLoader(eval_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    elif opt.mode == 'test':
        dataset_mode = 1
        print('>>> DATA loading >>>')
        dataset = datasets.Datasets(opt, mode='test')

        print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
        data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        # data_loader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    in_features = opt.in_features
    nb_kpts = int(in_features/3)  # number of keypoints


    print('>>> MODEL >>>')
    if opt.model == 'IAFormer':
        net_pred = model.IAFormer(seq_len=opt.seq_len, d_model=opt.d_model, opt=opt, num_kpt=nb_kpts, dataset="Mocap")

    net_pred.cuda()
    lr_now = opt.lr_now
    start_epoch = 1
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))

    if opt.mode == 'test':
        if '.pth.tar' in opt.ckpt:
            model_path_len = opt.ckpt
        elif opt.test_epoch is not None:
            model_path_len = '{}/ckpt_epo{}.pth.tar'.format(opt.ckpt, opt.test_epoch)
        else:
            model_path_len = './{}/ckpt_best.pth.tar'.format(opt.ckpt)

        print(">>> loading ckpt from '{}'".format(model_path_len))
        ckpt = torch.load(model_path_len)
        start_epoch = ckpt['epoch'] + 1
        lr_now = ckpt['lr']

        net_pred.load_state_dict(ckpt['state_dict'])
        
        print(">>> ckpt loaded (epoch: {} | err: {} | lr: {})".format(ckpt['epoch'], ckpt['err'], lr_now))


    if opt.mode == 'train': #train

        optimizer = optim.Adam(filter(lambda x: x.requires_grad, net_pred.parameters()), lr=opt.lr_now)

        util.save_ckpt({'epoch': 0, 'lr': lr_now, 'err': 0, 'state_dict': net_pred.state_dict(), 'optimizer': optimizer.state_dict()}, 0, dataset="Mocap",opt=opt)
        writer = SummaryWriter(opt.tensorboard)
        mpjpe_flag = 10000

        for epo in tqdm(range(start_epoch, opt.epoch + 1)):
            ret_train = run_model(nb_kpts, net_pred, opt.batch_size, optimizer, data_loader=data_loader, opt=opt, epo=epo)
            mpjpe_mean, mpjpe_avg, ape_mean, vim_mean = eval(opt, net_pred, eval_data_loader, nb_kpts, epo)
            # exit(0)
            writer.add_scalar('scalar/train', ret_train['loss_train'], epo)

            lr_now = util.lr_decay_mine(optimizer, lr_now, 0.1 ** (1 / 50))

            ret_log = np.array([epo, lr_now, mpjpe_mean])
            head = np.array(['epoch', 'lr', 'mpjpe_mean'])
            for k in ret_train.keys():
                ret_log = np.append(ret_log, [ret_train[k]])
                head = np.append(head, [k])
            util.save_csv_log(opt, head, ret_log, is_create=(epo == 1), file_name="train_log")
            if mpjpe_mean < mpjpe_flag:
                isbest = True
                mpjpe_flag = mpjpe_mean
            else:
                isbest = False

            print('epo{}, train error: {:.3f}, mpjpe_mean: {:.3f}, best_mean: {:.3f}, mpjpe_avg: {:.3f}, ape_mean: {:.3f}, vim_mean: {:.3f}  lr: {:.6f}'
                  .format(epo, ret_train['loss_train'], mpjpe_mean, mpjpe_flag, mpjpe_avg, ape_mean, vim_mean, lr_now))
            print(opt.ckpt)

            util.save_ckpt({'epoch': epo,
                            'lr': lr_now,
                            'err': ret_train['loss_train'],
                            'state_dict': net_pred.state_dict(),
                            'optimizer': optimizer.state_dict()},
                           epo, dataset="Mocap",opt=opt,Isbest=isbest)
        writer.close()

    else: #test

        run_model(nb_kpts, net_pred, opt.batch_size, data_loader=data_loader, opt=opt)

def run_model(nb_kpts, net_pred, batch_size, optimizer=None, data_loader=None, opt=None, epo=0):

    n = 0
    if opt.mode == 'train': #train
        net_pred.train()
        loss_train = 0
        for batch_idx, (x, y) in enumerate(data_loader): # in_n + kz

            torch.cuda.empty_cache()
            if np.shape(x)[0] < opt.batch_size:
                continue #when only one sample in this batch
            n += batch_size

            x = x.float().cuda()
            y = y.float().cuda()

            x_c = x.clone().detach()
            y_c = y.clone().detach()

            data_out, mix_loss = net_pred(x_c, y_c)

            data_gt = y_c.transpose(2, 3)
            loss = mix_loss

            optimizer.zero_grad()
            loss.backward()
            loss_train += loss.item() * batch_size
            optimizer.step()

        res_dic = {"loss_train" : loss_train / n }
        return res_dic

    else: #test
        net_pred.eval()
        mpjpe_joi = np.zeros([opt.seq_len])
        ape_joi = np.zeros([5])
        vim_joi = np.zeros([5])
        # n = 0
        for batch_idx, (x, y) in enumerate(data_loader): # raw_in_n + out_n
            if np.shape(x)[0] < opt.batch_size:
                continue #when only one sample in this batch
            n += batch_size

            x = x.float().cuda()
            y = y.float().cuda()
            # print(x.shape, y.shape)

            data_out, _ = net_pred(x, y)#[:,:,0]  # bz, 2kz, 108
            data_gt = y.transpose(2, 3)
            num_per = y.shape[1]


            data_gt = data_gt.reshape(opt.batch_size, num_per, opt.seq_len, nb_kpts, 3)
            data_out = data_out.reshape(opt.batch_size, num_per, opt.seq_len, nb_kpts, 3)
            tmp_joi = torch.sum(torch.mean(torch.mean(torch.norm(data_gt - data_out, dim=4), dim=3), dim=1), dim=0)
            # print(tmp_joi)
            mpjpe_joi += tmp_joi.cpu().data.numpy()

            tmp_ape_joi = APE(data_out[:, :, opt.frame_in:, :, :], data_gt[:, :, opt.frame_in:, :, :], [4, 9, 14, 19, 24])
            ape_joi += tmp_ape_joi#.data.numpy()

            data_vim_gt = data_gt[:, :, opt.frame_in:, :, :].transpose(2, 1)
            data_vim_gt = data_vim_gt.reshape(opt.batch_size, opt.seq_len, -1, 3)
            data_vim_pred = data_out[:, :, opt.frame_in:, :, :].transpose(2, 1)
            data_vim_pred = data_vim_pred.reshape(opt.batch_size, opt.seq_len, -1, 3)
            tmp_vim_joi = batch_VIM(data_vim_gt.cpu().data.numpy(), data_vim_pred.cpu().data.numpy(), [4, 9, 14, 19, 24])
            vim_joi += tmp_vim_joi#.data.numpy()



        mpjpe_joi = mpjpe_joi/n * 1000  # n = testing dataset length
        ape_joi = ape_joi/n * 1000 * opt.batch_size
        vim_joi = vim_joi/n * 100
        # print(ape_joi.shape, vim_joi.shape)
        select_frame = [4, 9, 14, 19, 24]
        print(mpjpe_joi[opt.frame_in:][select_frame])
        print("APE: ", ape_joi)
        print("VIM: ", vim_joi)

        mpjpe_mean = np.mean(mpjpe_joi[opt.frame_in:][select_frame])
        mpjpe_avg = np.mean(mpjpe_joi[opt.frame_in:])
        ape_mean = np.mean(ape_joi)
        vim_mean = np.mean(vim_joi)

        res_dic = {"mpjpe_joi": mpjpe_joi}


        if opt.save_results:
            import json
            key_exp = opt.exp + '_testepo'+str(opt.test_epoch)
            print('save name exp:', opt.exp)
            print('MPJPE mean: ', mpjpe_mean)
            print('MPJPE AVG: ', mpjpe_avg)
            print('APE_mean: ', ape_mean)
            print('VIM_mean: ', vim_mean)

            ts = "AGV"

            results = {key_exp: {}}
            results[key_exp][ts]={"mpjpe_joi": mpjpe_joi.tolist()}

            with open('{}/results.json'.format(opt.ckpt), 'w') as w:
                json.dump(results, w)

        return res_dic

def eval(opt, net_pred, data_loader, nb_kpts, epo):
    net_pred.eval()
    mpjpe_joi = np.zeros([opt.seq_len])
    ape_joi = np.zeros([5])
    vim_joi = np.zeros([5])
    n = 0

    for batch_idx, (x, y) in enumerate(data_loader): # in_n + kz
        if np.shape(x)[0] < opt.batch_size:
            continue #when only one sample in this batch
        n += opt.batch_size

        x = x.float().cuda()
        y = y.float().cuda()


        data_out, loss = net_pred(x, y)#[:,:,0]  # bz, 2kz, 108
        num_per = y.shape[1]
        data_gt = y.transpose(2, 3)


        # print(data_out.shape, data_gt.shape)
        data_gt = data_gt.reshape(opt.batch_size, num_per, opt.seq_len, nb_kpts, 3)
        data_out = data_out.reshape(opt.batch_size, num_per, opt.seq_len, nb_kpts, 3)
        tmp_joi = torch.sum(torch.mean(torch.mean(torch.norm(data_gt - data_out, dim=4), dim=3), dim=1), dim=0)
        # print(tmp_joi)
        mpjpe_joi += tmp_joi.cpu().data.numpy()

        tmp_ape_joi = APE(data_out[:, :, opt.frame_in:, :, :], data_gt[:, :, opt.frame_in:, :, :], [4, 9, 14, 19, 24])
        ape_joi += tmp_ape_joi#.data.numpy()

        data_vim_gt = data_gt[:, :, opt.frame_in:, :, :].transpose(2, 1)
        data_vim_gt = data_vim_gt.reshape(opt.batch_size, opt.seq_len, -1, 3)
        data_vim_pred = data_out[:, :, opt.frame_in:, :, :].transpose(2, 1)
        data_vim_pred = data_vim_pred.reshape(opt.batch_size, opt.seq_len, -1, 3)
        tmp_vim_joi = batch_VIM(data_vim_gt.cpu().data.numpy(), data_vim_pred.cpu().data.numpy(), [4, 9, 14, 19, 24])
        vim_joi += tmp_vim_joi#.data.numpy()

    mpjpe_joi = mpjpe_joi/n * 1000  # n = testing dataset length
    ape_joi = ape_joi/n * 1000 * opt.batch_size
    vim_joi = vim_joi/n * 1000
    print(ape_joi.shape, vim_joi.shape)
    print(mpjpe_joi)
    print("APE: ", ape_joi)
    print("VIM: ", vim_joi)
    select_frame = [4, 9, 14, 19, 24]
    mpjpe_mean = np.mean(mpjpe_joi[opt.frame_in:][select_frame])
    mpjpe_avg = np.mean(mpjpe_joi[opt.frame_in:])
    ape_mean = np.mean(ape_joi)
    vim_mean = np.mean(vim_joi)


    if opt.save_results:
        import json
        key_exp = 'epoch:'+str(epo)

        results = {key_exp: {}}
        results[key_exp]["mpjpe_joi"]=mpjpe_joi.tolist()
        results[key_exp]["mpjpe_mean"]=mpjpe_mean.tolist()
        results[key_exp]["ape_joi"]=ape_joi.tolist()
        results[key_exp]["ape_mean"]=ape_mean.tolist()
        results[key_exp]["vim_joi"]=vim_joi.tolist()
        results[key_exp]["vim_mean"]=vim_mean.tolist()
        # print(mpjpe_joi)
        with open('{}/eval_results.json'.format(opt.ckpt), 'a') as w:
            json.dump(results, w)
            w.write('\n')

    return mpjpe_mean, mpjpe_avg, ape_mean, vim_mean

def APE(V_pred, V_trgt, frame_idx):

    V_pred = V_pred - V_pred[:, :, :, 0:1, :]
    V_trgt = V_trgt - V_trgt[:, :, :, 0:1, :]

    err = np.arange(len(frame_idx), dtype=np.float_)

    for idx in range(len(frame_idx)):
        err[idx] = torch.mean(torch.mean(torch.norm(V_trgt[:, :, frame_idx[idx]-1, :, :] - V_pred[:, :, frame_idx[idx]-1, :, :], dim=3), dim=2),dim=1).cpu().data.numpy().mean()
    return err

def batch_VIM(GT, pred, select_frames):
    '''Calculate the VIM at selected timestamps.

    Args:
        GT: [B, T, J, 3].

    Returns:
        errorPose: [T].
    '''
    errorPose = np.power(GT - pred, 2)
    errorPose = np.sum(errorPose, axis=(2, 3))
    errorPose = np.sqrt(errorPose)
    errorPose = errorPose.sum(axis=0)
    # scale = 100
    return errorPose[select_frames]# * scale

if __name__ == '__main__':
    option = Options().parse()
    main(option)


