# Level Perceptron
import torch

def Trajectory_Weight(opt, skeleton):
    num_person = skeleton.shape[0]
    # print(skeleton.shape)  # [3, 45, 75]
    center_co = torch.mean(skeleton[:, :3, opt.frame_in-1], dim=0)
    # print(center_co.shape, center_co.unsqueeze(1).repeat(1, 50).shape, skeleton[:, :3, :50].shape)
    move_score = torch.ones(num_person)

    for i in range(num_person):
        move_score[i] = torch.norm(skeleton[i, :3, :opt.frame_in]-center_co.unsqueeze(1).repeat(1, opt.frame_in))

    lp_score = LP_score(move_score)
    return lp_score.cuda()

def LP_score(move_score):
    num_person = len(move_score)
    lp_score = torch.ones(num_person)

    for i in range(num_person):
        lp_score[i] = torch.log(move_score[i]/torch.sum(move_score)+1)
    
    return lp_score