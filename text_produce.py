import os

import numpy as np

edge_num = 10
PATH = "D:\\MEC\\MEC\\data\\beijing\\07"  # 文件夹目录
e_l = np.zeros((edge_num, 2))
# calculate the mean of the data
path = PATH
files = os.listdir(path)  # 读入文件夹
num_txt = len(files)  # 统计文件夹中的文件个数
for num in range(num_txt):
    file_path = path + "\\" + files[num]
    file_path_save = path + "\\" + "save" + "\\"
    f = open(file_path, "r")
    f1 = f.readlines()
    # get line_num and initial data
    line_num = 0
    for line in f1:
        line_num += 1
    data = np.zeros((line_num, 2))
    index = 0
    for line in f1:
        data[index][0] = np.float(line.split(",")[2])  # x   以空格分隔符来取数据，将后面的两个数据保存到数组，， 取出来的数据是位置数据，（x,y）
        data[index][1] = np.float(line.split(",")[3])  # y
        print("x{},y{} position".format(data[index][0], data[index][1]))
        index += 1
