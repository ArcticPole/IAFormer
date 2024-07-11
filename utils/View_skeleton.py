import cv2
import numpy as np
import glob
import os

import json
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['image.interpolation'] = 'nearest'
# colors = plt.cm.hsv(np.linspace(0, 1, 10)).tolist()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

def view_Pa(opt, data_gt, data_out, index, mpjpe_temp):
    num_person = data_gt.shape[0]
    seq_len = data_out.shape[1]

    # Select HD Image index
    hd_idx = 400

    '''
    ## Visualize 3D Body
    '''
    # Edges between joints in the body skeleton
    # body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1

    body_edges = np.array([[0,1],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14]])

    data_gt = data_gt.cpu()
    data_out = data_out.cpu().detach().numpy()
    for frame_idx in range(seq_len):
        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev = -90, azim=-90)
        #ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        ax.axis('auto')
        # print(colors)
        # print(colors[0])
        for person_idx in range(num_person):
            for edge in body_edges:
                # print(person_idx)
                ax.plot(data_gt[person_idx, frame_idx, edge, 0],
                        data_gt[person_idx, frame_idx, edge, 1],
                        data_gt[person_idx, frame_idx, edge, 2],
                        color=colors[person_idx])
        for person_idx in range(num_person):
            for edge in body_edges:
                # print(person_idx)
                ax.plot(data_out[person_idx, frame_idx, edge, 0],
                        data_out[person_idx, frame_idx, edge, 1],
                        data_out[person_idx, frame_idx, edge, 2],
                        color=colors[num_person + person_idx])
        # plt.show()
        plt.title("batch{0}_frame{1}_MPJPE={2}".format(index, frame_idx, mpjpe_temp[frame_idx]))
        if not os.path.exists("{}/outputs".format(opt.ckpt)):
            os.makedirs("{}/outputs".format(opt.ckpt))
        fig.savefig("{0}/outputs/batch{1}_frame{2}.jpg".format(opt.ckpt, index, frame_idx))
        plt.close(fig)
        # exit(0)

def view_ch(opt, data_gt, data_out, index, mpjpe_temp):
    num_person = data_gt.shape[0]
    seq_len = data_out.shape[1]

    # Select HD Image index
    hd_idx = 400

    '''
    ## Visualize 3D Body
    '''
    # Edges between joints in the body skeleton
    # body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1

    body_edges = np.array([[10, 9], [9, 8], [8, 11], [8, 14], [11, 12], [14, 15], [12, 13], [15, 16],
                           [8, 7], [7, 0], [0, 1], [0, 4], [1, 2], [4, 5], [2, 3], [5, 6],
                           [13, 21], [13, 22], [16, 23], [16, 24], [3, 17], [3, 18], [6, 19], [6, 20]])

    data_gt = data_gt.cpu()
    data_out = data_out.cpu().detach().numpy()
    for frame_idx in range(seq_len):
        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev = -90, azim=-90)
        #ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        ax.axis('auto')
        # print(colors)
        # print(colors[0])
        for person_idx in range(num_person):
            for edge in body_edges:
                # print(person_idx)
                ax.plot(data_gt[person_idx, frame_idx, edge, 0],
                        data_gt[person_idx, frame_idx, edge, 1],
                        data_gt[person_idx, frame_idx, edge, 2],
                        color=colors[person_idx])
        for person_idx in range(num_person):
            for edge in body_edges:
                # print(person_idx)
                ax.plot(data_out[person_idx, frame_idx, edge, 0],
                        data_out[person_idx, frame_idx, edge, 1],
                        data_out[person_idx, frame_idx, edge, 2],
                        color=colors[num_person + person_idx])
        # plt.show()
        plt.title("batch{0}_frame{1}_MPJPE={2}".format(index, frame_idx, mpjpe_temp[frame_idx]))
        if not os.path.exists("{}/outputs".format(opt.ckpt)):
            os.makedirs("{}/outputs".format(opt.ckpt))
        fig.savefig("{0}/outputs/batch{1}_frame{2}.jpg".format(opt.ckpt, index, frame_idx))
        plt.close(fig)
    video(opt, 'all')
        # exit(0)

def view_hm(opt, data_gt, data_out, index, mpjpe_temp):
    seq_len = data_out.shape[1]
    body_edges = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[0,6],[6,7],[7,8],[8,9],[9,10],
                           [0,12],[12,13],[13,14],[14,15],[13,17],[17,18],[18,19],[13,25],
                           [25,26],[26,27]])
    # data_gt = data_gt.cpu()
    # data_out = data_out.cpu().detach().numpy()
    for frame_idx in range(seq_len):
        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev = -90, azim=-90)
        #ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        ax.axis('auto')
        # print(colors)
        # print(colors[0])

        # for edge in body_edges:
        #     # print(person_idx)
        #     ax.plot(data_gt[frame_idx, edge, 0],
        #             data_gt[frame_idx, edge, 1],
        #             data_gt[frame_idx, edge, 2],
        #             color=colors[0])
        for edge in body_edges:
            # print(person_idx)
            ax.plot(data_out[frame_idx, edge, 0],
                    data_out[frame_idx, edge, 1],
                    data_out[frame_idx, edge, 2],
                    color=colors[1])
        # plt.show()

        plt.title("batch{0}_frame{1}_MPJPE={2}".format(index, frame_idx, mpjpe_temp[frame_idx]))
        if not os.path.exists("{}/outputs".format(opt.ckpt)):
            os.makedirs("{}/outputs".format(opt.ckpt))
        fig.savefig("{0}/outputs/batch{1}_frame{2}.jpg".format(opt.ckpt, index, frame_idx))
        plt.close(fig)

def view_mocap(opt, data_gt, data_out, index, mpjpe_temp):
    print(data_gt.shape,data_out.shape)
    num_person = data_gt.shape[0]
    seq_len = data_out.shape[1]

    # Select HD Image index
    hd_idx = 400

    '''
    ## Visualize 3D Body
    '''
    # Edges between joints in the body skeleton
    # body_edges = np.array([[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[8,12],[12,13],[13,14],[8,15]])-1
    body_edges = np.array([[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[8,10],[10,11],[11,12],[8,13],[13,14],[14,15]])-1
    body_joints = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-1


    data_gt = data_gt.cpu()
    data_out = data_out.cpu().detach().numpy()
    # plt.grid(b=None)

    for frame_idx in range(seq_len):
        # from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        # ax.grid(b=None)
        # ax.view_init(elev = -90, azim=-90)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.axis('auto')
        # print(colors)
        # print(colors[0])
        for person_idx in range(num_person):
            for edge in body_edges:
                # print(person_idx)
                ax.plot(data_gt[person_idx, frame_idx, edge, 0],
                        data_gt[person_idx, frame_idx, edge, 2],
                        data_gt[person_idx, frame_idx, edge, 1],
                        color=colors[person_idx])
            for joint in body_joints:
                ax.plot(data_gt[person_idx, frame_idx, joint, 0],
                            data_gt[person_idx, frame_idx, joint, 2],
                            data_gt[person_idx, frame_idx, joint, 1],
                            color='black', marker='o', markersize=4)
        if not os.path.exists("{}/outputs".format(opt.ckpt)):
            os.makedirs("{}/outputs".format(opt.ckpt))
        fig.savefig("{0}/outputs/gt_batch{1}_frame{2}.png".format(opt.ckpt, index, frame_idx))
        plt.close(fig)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        for person_idx in range(num_person):
            for edge in body_edges:
                # print(person_idx)
                ax.plot(data_out[person_idx, frame_idx, edge, 0],
                        data_out[person_idx, frame_idx, edge, 2],
                        data_out[person_idx, frame_idx, edge, 1],
                        # color=colors[num_person + person_idx])
                        color=colors[person_idx])
            for joint in body_joints:
                ax.plot(data_out[person_idx, frame_idx, joint, 0],
                        data_out[person_idx, frame_idx, joint, 2],
                        data_out[person_idx, frame_idx, joint, 1],
                        color='black', marker='o', markersize=4)
        # plt.show()
        # plt.title("batch{0}_frame{1}_MPJPE={2}".format(index, frame_idx, mpjpe_temp[frame_idx]))
        if not os.path.exists("{}/outputs".format(opt.ckpt)):
            os.makedirs("{}/outputs".format(opt.ckpt))
        fig.savefig("{0}/outputs/pred_batch{1}_frame{2}.png".format(opt.ckpt, index, frame_idx))
        plt.close(fig)
    # video(opt, 'all')
        # exit(0)


def view_sequence(opt, data_gt, data_out):
    num_person = data_gt.shape[0]
    seq_len = data_out.shape[1]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    '''
    ## Visualize 3D Body
    '''

    body_edges = np.array([[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[8,10],[10,11],[11,12],[8,13],[13,14],[14,15]])-1
    body_joints = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])-1


    data_gt = data_gt.cpu()
    data_out = data_out.cpu().detach().numpy()
    # plt.grid(b=None)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    al_index = 0
    for frame_idx in range(0, opt.frame_in, 10):
        # from mpl_toolkits.mplot3d import Axes3D
        for person_idx in range(num_person):
            for edge in body_edges:
                # print(person_idx)
                ax.plot(data_gt[person_idx, frame_idx, edge, 0],
                        data_gt[person_idx, frame_idx, edge, 2],
                        data_gt[person_idx, frame_idx, edge, 1],
                        color=colors[person_idx], linewidth = 4, alpha=alphas[al_index])
            for joint in body_joints:
                ax.plot(data_gt[person_idx, frame_idx, joint, 0],
                            data_gt[person_idx, frame_idx, joint, 2],
                            data_gt[person_idx, frame_idx, joint, 1],
                            color='black', linewidth = 4, marker='o', markersize=5, alpha=alphas[al_index])
        al_index += 1

    if not os.path.exists("{}/outputs".format(opt.ckpt)):
        os.makedirs("{}/outputs".format(opt.ckpt))
    fig.savefig("{0}/outputs/sequence_historical.png".format(opt.ckpt))
    plt.close(fig)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    al_index = 0
    for frame_idx in range(opt.frame_in, opt.seq_len, 5):
        # from mpl_toolkits.mplot3d import Axes3D
        for person_idx in range(num_person):
            for edge in body_edges:
                # print(person_idx)
                ax.plot(data_gt[person_idx, frame_idx, edge, 0],
                        data_gt[person_idx, frame_idx, edge, 2],
                        data_gt[person_idx, frame_idx, edge, 1],
                        color=colors[person_idx], linewidth = 4, alpha=alphas[al_index])
            for joint in body_joints:
                ax.plot(data_gt[person_idx, frame_idx, joint, 0],
                            data_gt[person_idx, frame_idx, joint, 2],
                            data_gt[person_idx, frame_idx, joint, 1],
                            color='black', linewidth = 4, marker='o', markersize=5, alpha=alphas[al_index])
        al_index += 1

    if not os.path.exists("{}/outputs".format(opt.ckpt)):
        os.makedirs("{}/outputs".format(opt.ckpt))
    fig.savefig("{0}/outputs/sequence_future.png".format(opt.ckpt))
    plt.close(fig)
    # video(opt, 'all')
        # exit(0)


def view_root(opt, data_gt, data_out):
    num_person = data_gt.shape[0]
    seq_len = data_out.shape[1]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    '''
    ## Visualize 3D Body
    '''

    body_joints = np.array([1])-1
    data_gt = data_gt.cpu()
    data_out = data_out.cpu().detach().numpy()
    # plt.grid(b=None)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    
    al_index = 0
    for frame_idx in range(0, opt.seq_len, 10):
        # from mpl_toolkits.mplot3d import Axes3D
        for person_idx in range(num_person):

            for joint in body_joints:
                ax.plot(data_gt[person_idx, frame_idx, joint, 0],
                        data_gt[person_idx, frame_idx, joint, 2],
                        # data_gt[person_idx, frame_idx, joint, 1],
                        color=colors[person_idx], linewidth = 4, marker='o', markersize=5, alpha=alphas[al_index])
        al_index += 1

    if not os.path.exists("{}/outputs".format(opt.ckpt)):
        os.makedirs("{}/outputs".format(opt.ckpt))
    fig.savefig("{0}/outputs/sequence_root.png".format(opt.ckpt))
    plt.close(fig)


def video(opt, mode='all'):
    # 其它格式的图片也可以
    img_array = []
    filenames = glob.glob("{}/outputs/*.png".format(opt.ckpt))
    filenames.sort(key=lambda x: (len(x), x[-6], x[-5]))
    print(filenames)
    for filename in filenames:

        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    # avi：视频类型，mp4也可以
    # cv2.VideoWriter_fourcc(*'DIVX')：编码格式
    # 5：视频帧率
    # size:视频中图片大小t

    out = cv2.VideoWriter("{}/outputs/{}.mp4".format(opt.ckpt, mode),
                          cv2.VideoWriter_fourcc(*'mp4v'),
                        #   cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                          1, size)


    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
