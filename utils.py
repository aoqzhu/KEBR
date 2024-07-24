# -*-coding:utf-8-*- 
import numpy as np
import os
import random
import torch
from thop.profile import profile
from thop import clever_format
import xlwt

# set random seed
def set_random_seed(seed):
    print("Random Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num


def get_flops(model, *input):
    macs, num_params = profile(model, input, verbose=False)
    macs, num_params = clever_format([macs, num_params], '%.3f')
    return macs, num_params

def interval_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def file_save(e_list,t_list,a_list,v_list,file_name):
    i =0
    file = xlwt.Workbook('encoding = utf-8')
    sheet1 = file.add_sheet('sheet1', cell_overwrite_ok=True)
    sheet1.write(0, 0, "序号")
    sheet1.write(0, 1, "e_loss")
    sheet1.write(0, 2, "t_loss")
    sheet1.write(0, 3, "a_loss")
    sheet1.write(0, 4, "v_loss")
    for i in range(len(a_list)):
        sheet1.write(i + 1, 0, i)
        sheet1.write(i + 1, 1, e_list[i])
        sheet1.write(i + 1, 2, t_list[i])
        sheet1.write(i + 1, 3, a_list[i])
        sheet1.write(i + 1, 4, v_list[i])
    file.save(file_name+'.xls')
