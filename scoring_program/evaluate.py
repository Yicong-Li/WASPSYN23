#Generates a score.txt file containing pre-, post and final scores
import h5py
import numpy as np
import os
from sys import argv
from glob import glob
from scipy.optimize import linear_sum_assignment

def ls(filename):
    return (sorted(glob(filename)))

if (os.name == "nt"):
    filesep = '\\'
else:
    filesep = '/'

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
    assignments = dict(zip(final_row_ind, final_col_ind))
    associated_cost = cost_m[final_row_ind, final_col_ind].sum()

    fp_fn_1 = np.abs(test.shape[0] - gt.shape[0])
    fp_fn_2 = (len(list(zip(row_ind, col_ind))) - len(list(zip(final_row_ind, final_col_ind)))) * 2
    tp = len(list(zip(final_row_ind, final_col_ind)))
    fscore = 2 * tp / (2 * tp + fp_fn_1 + fp_fn_2)
    return assignments, associated_cost, fscore, final_row_ind, final_col_ind

def score(input_dir, output_dir):
    
    total_sum = 0
    submit_dir = os.path.join(input_dir, 'res') 
    #submission directory ./test_sample%d_vol%d/test_sample%d_vol%d_pred.h5
    truth_dir = os.path.join(input_dir, 'ref')
    #reference ground truth directory ./test_sample%d_vol%d/syns_zyx_5500-6100_6000-6600_1800-2400.h5

    #|- input
    #|- ref (This is the reference data unzipped)
    #|- res (This is the user submission unzipped)

    if not os.path.isdir(submit_dir):
        print(submit_dir, " doesn't exist")
    
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = os.path.join(output_dir, 'scores.txt')
        output_file = open(output_filename, 'w')

        solution_folder_names = [f"test_sample{i + 1}_vol{j}" for i in range(3) for j in range(3)] # list of file names

        for i, solution_folder in enumerate(solution_folder_names):
            
            gt_file= ls(os.path.join(truth_dir, solution_folder,'*.h5'))[-1]
            predict_file = ls(os.path.join(submit_dir, solution_folder,'*predict.h5'))[-1]
            print("For test_file ", predict_file)
            if (predict_file == []): raise IOError('Missing prediction file')

            # Read the solution and prediction values
            f_gt = h5py.File(gt_file, 'r')
            f_test = h5py.File(predict_file, 'r')

            pre_gt = f_gt['pre'][:]
            post_gt = f_gt['post'][:]
            pre_test = f_test['pre'][:]
            post_test = f_test['post'][:]

            f_gt.close()
            f_test.close()

            # offset gt
            #path_gt = 'datasets/training_set/train_sample3_vol1/syns_zyx_3680-4096_2944-3360_4448-4864.h5'
            offset_zyx = gt_file.split('/')[-1].split('_')
            offset_z = float(offset_zyx[2].split('-')[0])
            offset_y = float(offset_zyx[3].split('-')[0])
            offset_x = float(offset_zyx[4].split('-')[0])
            pre_gt = pre_gt - [offset_z, offset_y, offset_x]
            post_gt[:, 1:] = post_gt[:, 1:] - [offset_z, offset_y, offset_x]
#             pre_test = pre_test - [offset_z, offset_y, offset_x]
#             post_test[:, 1:] = post_test[:, 1:] - [offset_z, offset_y, offset_x]


            # evaluation pre
            pre_assignments, pre_associated_cost, pre_fscore, pre_test_node, pre_gt_node = assign_cal_f1(pre_test, pre_gt, 11, True)
            # print('pre_assignments:', pre_assignments)
            # print('pre_cost:', pre_associated_cost)
            print("pre_fscore: %0.12f\n" % pre_fscore)

            # evaluation post
            post_fscore_all = []
            for i in list(zip(pre_test_node, pre_gt_node)):
                post_test_each = post_test[post_test[:, 0] == i[0]]
                post_gt_each = post_gt[post_gt[:, 0] == i[1]]
                if len(post_test_each[:, 1:]) == 0 and len(post_gt_each[:, 1:]) == 0:
                    continue
                post_assignments_each, post_associated_cost_each, post_fscore_each, _, _ = assign_cal_f1(post_test_each[:,1:], post_gt_each[:, 1:], 6.5, True)
                post_fscore_all.append(post_fscore_each)
            if post_fscore_all == []:
                post_fscore=0
            else:
                post_fscore = np.mean(post_fscore_all)
            print("post_fscore: %0.12f\n" % post_fscore)

            # evaluation all
            final_fscore = 0.5 * pre_fscore + 0.5 * post_fscore
            print("final_fscore: %0.12f\n" % final_fscore)

            total_sum += final_fscore

        output_file.write("correct: %0.12f\n" % (total_sum/9.0))
        output_file.close()

if __name__ == "__main__":

    #The scoring program will be invoked as <program> <input directory> <output directory>
    #### INPUT/OUTPUT: Get input and output files
    input_dir = argv[1]
    output_dir= argv[2]

    score(input_dir, output_dir)
