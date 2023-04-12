import math
import os

import numpy as np

edge_num = 10
PATH = "D:\\MEC\\MEC\\data\\beijing\\06"  # 文件夹目录
e_l = np.zeros((edge_num, 2))
# calculate the mean of the data
path = PATH
files = os.listdir(path)  # 读入文件夹
num_txt = len(files)  # 统计文件夹中的文件个数
group_num = math.floor(num_txt / edge_num)  # 向下取整
edge_id = 0
for base in range(0, group_num * edge_num, group_num):
    for data_num in range(base, base + group_num):
        # data_name = str("%03d" % (data_num + 1))  # plus zero
        # file_name = LOCATION + "_30sec_" + data_name + ".txt"
        file_path = path + "\\" + files[data_num]
        f = open(file_path, "r")
        f1 = f.readlines()
        # get line_num and initial data
        line_num = 0
        for line in f1:
            line_num += 1
        data = np.zeros((line_num, 2))
        # collect the data from the .txt
        index = 0
        for line in f1:
            data[index][0] = np.float(line.split(",")[2])  # x   以空格分隔符来取数据，将后面的两个数据保存到数组，， 取出来的数据是位置数据，（x,y）
            data[index][1] = np.float(line.split(",")[3])  # y
            index += 1
        # stack the collected data
        if data_num % group_num == 0:
            cal = data
        else:
            cal = np.vstack((cal, data))  # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
    e_l[edge_id] = np.mean(cal,
                           axis=0)  # axis=0，那么输出矩阵是1行，求每一列的平均（按照每一行去求平均）；axis=1，输出矩阵是1列，求每一行的平均（按照每一列去求平均）。还可以这么理解，axis是几，那就表明哪一维度被压缩成1。
    edge_id += 1

print(e_l)
