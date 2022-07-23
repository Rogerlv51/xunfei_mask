
import os
import pandas as pd

def make_label():
    originpath = os.getcwd()    # 返回根地址
    traindir = os.listdir("./")[7]
    label0 = "mask_weared_incorrect"
    label1 = "with_mask"
    label2 = "without_mask"
    path0 = originpath + "\\" + traindir + "\\" + label0
    path1 = originpath + "\\" + traindir + "\\" + label1
    path2 = originpath + "\\" + traindir + "\\" + label2
    incorrect_namelist = os.listdir(path0)
    mask_namelist = os.listdir(path1)
    without_namelist = os.listdir(path2)
    data1 = {"image": incorrect_namelist, "label": label0}
    data1 = pd.DataFrame(data=data1)
    data2 = {"image": mask_namelist, "label": label1}
    data2 = pd.DataFrame(data=data2)
    data3 = {"image": without_namelist, "label": label2}
    data3 = pd.DataFrame(data=data3)
    data = pd.concat([data1, data2, data3], axis=0)     # 上下合并
    return data


data = make_label()

