import matplotlib.pyplot as plt
import pandas as pd

def draw_result(opt, mpjpe=None):
    # reference_path = "/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/Third-part/PGBIG/checkpoint/main_h36m_3d_all_in10_out25_ks10_dctn35_dropout_0.3_lr_0.005_d_model_16_e_6_d_6/test_all_eval.csv"
    reference_path = "/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/Third-part/PGBIG/checkpoint_xiao/main_h36m_3d_all_in10_out25_ks10_dctn35_dropout_0.3_lr_0.005_d_model_16_e_6_d_6/test_all_eval.csv"
    re_data = pd.read_csv(reference_path).values[-1][1:]
    mpjpe = mpjpe[10:]
    index = range(len(re_data))
    plt.plot(index, re_data, 'b', label='reference(STAGE_4)')
    plt.plot(index, mpjpe, 'r', label='ours(MultiRes)')
    plt.xlabel("frame")
    plt.ylabel("MPJPE")
    plt.legend()
    plt.title('MultiRes vs state_4 on Human3.6M')
    plt.savefig('{}/compare.jpg'.format(opt.ckpt))
    plt.show()

def draw_result_chi(opt, mpjpe_name_1=None, mpjpe_name_2=None):
    import json

    mpjpe_path_1 = '/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/{}/CHI_results.json'.format(mpjpe_name_1)
    mpjpe_path_2 = '/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/{}/CHI_results.json'.format(mpjpe_name_2)

    f1 = open(mpjpe_path_1, 'r')
    content1 = f1.read()
    a = json.loads(content1)

    f2 = open(mpjpe_path_2, 'r')
    content2 = f2.read()
    b = json.loads(content2)

    mpjpe_kind_1 = '{}_testepoNone'.format(mpjpe_name_1)
    mpjpe_kind_2 = '{}_testepoNone'.format(mpjpe_name_2)

    mpjpe_1 = a[mpjpe_kind_1]["AGV"]["mpjpe_joi"][10:]
    mpjpe_2 = b[mpjpe_kind_2]["AGV"]["mpjpe_joi"][10:]

    print(mpjpe_1)
    index = range(len(mpjpe_1))
    plt.plot(index, mpjpe_1, 'r', label='ours(MultiRes)')
    plt.plot(index, mpjpe_2, 'b', label='reference(STAGE_4)')
    plt.xlabel("frame")
    plt.ylabel("MPJPE")
    plt.legend()
    plt.title('MultiRes vs state_4 on CHI3D')
    plt.savefig('{}/MultiRes vs STAGE_4 on CHI3D.jpg'.format(opt.ckpt))
    plt.show()

def draw_train_log(path_1, path_2, path_3, path_4, save_name):

    # data_1 = pd.read_csv(path_1).values[:, 2]
    # data_2 = pd.read_csv(path_2).values[:, 2]
    # data_3 = pd.read_csv(path_3).values[:, 2]
    # data_4 = pd.read_csv(path_4).values[:, 2]

    data_1 = pd.read_csv(path_1).values[30:49, 2]
    data_2 = pd.read_csv(path_2).values[30:49, 2]
    data_3 = pd.read_csv(path_3).values[30:49, 2]
    data_4 = pd.read_csv(path_4).values[30:49, 2]

    index = range(len(data_1))
    plt.plot(index, data_1, 'b', label='usIP&TC_True&Ture')
    plt.plot(index, data_2, 'g', label='usIP&TC_Ture&False')
    plt.plot(index, data_3, 'y', label='usIP&TC_False&True')
    plt.plot(index, data_4, 'r', label='usIP&TC_False&False')
    # plt.plot(index, mpjpe, 'r', label='ours(MultiRes)')
    plt.xlabel("frame")
    plt.ylabel("MPJPE")
    plt.legend()
    plt.title('Ablation result of Model')
    plt.savefig(save_name)
    plt.show()


# draw_train_log(path_1='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/exp_CHI3D_changeRes_IP_MultiRes_in10_out25_usIP&TC_True&True_lr_0.01_lrd_0.98_d_model_256/CHI_train_log.csv',
#                path_2='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/exp_CHI3D_changeRes_IP_MultiRes_in10_out25_usIP&TC_True&False_lr_0.01_lrd_0.98_d_model_256/CHI_train_log.csv',
#                path_3='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/exp_CHI3D_changeRes_IP_MultiRes_in10_out25_usIP&TC_False&True_lr_0.01_lrd_0.98_d_model_256/CHI_train_log.csv',
#                path_4='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/exp_CHI3D_changeRes_IP_MultiRes_in10_out25_usIP&TC_False&False_lr_0.01_lrd_0.98_d_model_256/CHI_train_log.csv',
#                save_name='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/outputs/ablation_view/CHI_ablation.jpg')

# draw_train_log(path_1='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/exp_Human3.6M_MultiRes_in10_out25_usIP&TC_True&True_lr_0.01_lrd_0.98_d_model_256/H36M_train_log.csv',
#                path_2='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/exp_Human3.6M_MultiRes_in10_out25_usIP&TC_True&False_lr_0.01_lrd_0.98_d_model_256/H36M_train_log.csv',
#                path_3='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/exp_Human3.6M_MultiRes_in10_out25_usIP&TC_False&True_lr_0.01_lrd_0.98_d_model_256/H36M_train_log.csv',
#                path_4='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/model/checkpoint/exp_Human3.6M_MultiRes_in10_out25_usIP&TC_False&False_lr_0.01_lrd_0.98_d_model_256/H36M_train_log.csv',
#                save_name='/media/ub/d68617c9-6629-41d9-a9a2-4f61b5cad92b/xiaopeng/HPF_dev1/outputs/ablation_view/H36M_ablation(30-50).jpg')
# draw_result()
# draw_result_chi(opt,
#                 "exp_CHI3D_changeRes_IP_MultiRes_in10_out25_usIP&TC_True&True_lr_0.01_lrd_0.98_d_model_256_testepo50",
#                 "exp_CHI3D_changeRes_IP_stage_4_in10_out25_usIP&TC_True&True_lr_0.01_lrd_0.98_d_model_256_testepo50")