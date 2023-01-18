import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

# calculate cost: l2-norm
def cost_matrix(test, gt):
    matrix = np.zeros((test.shape[0], gt.shape[0]))
    for i in range(gt.shape[0]):
        diff = test - gt[i, :]
        square = np.power(diff, 2)
        sum_of_square = np.sum(square, axis=1)
        cost = np.sqrt(sum_of_square)
        matrix[:, i] = cost
    return matrix

# assign and calculate f-score
def assign_cal_f1(test, gt, radius, use_radius):
    cost_m = cost_matrix(test, gt)

    row_ind, col_ind = linear_sum_assignment(cost_m)
    final_row_ind = list(row_ind)
    final_col_ind = list(col_ind)
    if use_radius:
        r = radius
        for i in list(zip(row_ind, col_ind)):
            if cost_m[i[0], i[1]] > r:
                final_row_ind.remove(i[0])
                final_col_ind.remove(i[1])
    test_id = np.arange(test.shape[0])[final_row_ind]
    gt_id = np.arange(gt.shape[0])[final_col_ind]
    assignments = dict(zip(test_id, gt_id))
    associated_cost = cost_m[final_row_ind, final_col_ind].sum()

    fp_fn_1 = np.abs(test.shape[0] - gt.shape[0])
    fp_fn_2 = (len(list(zip(row_ind, col_ind))) - len(list(zip(final_row_ind, final_col_ind)))) * 2
    tp = len(list(zip(final_row_ind, final_col_ind)))
    fscore = 2 * tp / (2 * tp + fp_fn_1 + fp_fn_2)
    return assignments, associated_cost, fscore

# read data
path_gt = 'datasets/training_set/train_sample3_vol1/syns_zyx_3680-4096_2944-3360_4448-4864.h5'
path_test = 'datasets/training_set/train_sample3_vol0/syns_zyx_2217-2617_4038-4448_6335-6735.h5'
f_gt = h5py.File(path_gt, 'r')
f_test = h5py.File(path_test, 'r')
pre_gt = f_gt['pre'][:]
pre_test = f_test['pre'][:]
f_gt.close
f_test.close

# evaluation
pre_assignments, pre_associated_cost, pre_fscore = assign_cal_f1(pre_test, pre_gt, 2600, True)
print('pre_assignments:', pre_assignments)
print('pre_cost:', pre_associated_cost)
print('pre_fscore:', pre_fscore)