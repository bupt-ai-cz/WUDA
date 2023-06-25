import numpy as np
from scipy.stats import wasserstein_distance
import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description='path')
    parser.add_argument('--representationA_path', type=str, default='/root/representation_shift/checkpoints/GTAV_source/dataset_A',help='path to source domain representation')
    parser.add_argument('--representationB_path', type=str, default='/root/representation_shift/checkpoints/GTAV_source/dataset_B', help='path to target domain representation')
    opt = parser.parse_args()
    return opt

opt = parse_opt()

def representation_shift(dataset_A, dataset_B):
    '''
    dataset_A: array[channels, number]
    dataset_B: array[channels, number]
    number: number of samples
    '''
    sum_w_distance = 0
    for i in range(dataset_A.shape[0]):
        w_distance = wasserstein_distance(dataset_A[i,:],dataset_B[i,:])
        sum_w_distance = sum_w_distance + w_distance
    average_w_distance = sum_w_distance/dataset_A.shape[0]

    return average_w_distance

representationA_path = opt.representationA_path
representationB_path = opt.representationB_path

dataset_A = []
dataset_A = np.load(representationA_path)
dataset_B = []
dataset_B = np.load(representationB_path)

value = representation_shift(dataset_A,dataset_B)
print(value)