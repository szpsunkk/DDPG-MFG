##设置全部数据，不输出省略号
import sys

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
# path = './outputs/Vehicle edge computing (VEC) for digital twin/20220527-211756/results/'

boxes = np.load('./outputs/Vehicle edge computing (VEC) for digital twin/20220626-145031/results/train_rewards.npy')
print(boxes)
np.savetxt('./outputs/Vehicle edge computing (VEC) for digital twin/20220626-145031/results/train_rewards.txt', boxes,
           fmt='%s',
           newline='\n')
print('---------------------boxes--------------------------')
