import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

# read data
path = 'datasets/training_set/train_sample3_vol1/syns_zyx_3680-4096_2944-3360_4448-4864.h5'
f = h5py.File(path, 'r')
pre_gt = f['pre'][:]
f.close

# submission
pre_test = np.random.randint(1, 1000, (160, 3))

# calculate cost: l2-norm
pre_cost_matrix = np.zeros((pre_test.shape[0], pre_gt.shape[0]))
for i in range(pre_gt.shape[0]):
    diff = pre_test - pre_gt[i, :]
    square = np.power(diff, 2)
    sum_of_square = np.sum(square, axis=1)
    cost = np.sqrt(sum_of_square)
    pre_cost_matrix[:, i] = cost

# assignments
row_ind, col_ind = linear_sum_assignment(pre_cost_matrix)
pre_test_id = np.arange(pre_test.shape[0])[row_ind]
pre_gt_id = np.arange(pre_gt.shape[0])[col_ind]
assignments = dict(zip(pre_test_id, pre_gt_id))
associated_cost = pre_cost_matrix[row_ind, col_ind].sum()
fp_and_fn = np.abs(pre_test.shape[0] - pre_gt.shape[0])
print('assignments:', assignments)
print('cost:', associated_cost)
print('count of false negative & false positive:', fp_and_fn)
print(row_ind)