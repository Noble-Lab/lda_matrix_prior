from lda import *
import numpy as np 
import pandas as pd
import pickle
import argparse

xsi = 4000
n_vocab = 8000
n_topic = 30
beta = .1
alpha = .3
beta_ref = 1000
true_topic_word = None

parser = argparse.ArgumentParser(description='Make priors from the reference data')
parser.add_argument("input_wt", help="LDA MaxLikelihoodWordTopic to turn into a prior")
parser.add_argument("output", help="Output name")
parser.add_argument("topics", help="Number of topics", type=int)
args = parser.parse_args()

tw_pred_ref = np.loadtxt(open(args.input_wt, "r"), delimiter=",")
tw_pred_ref = tw_pred_ref.T + 1e-2
if args.topics == 1:
    tw_pred_ref = tw_pred_ref / sum(tw_pred_ref)
    tw_pred_ref = tw_pred_ref[np.newaxis, :]
else:
    normalize_matrix_by_row(tw_pred_ref) 
np.savetxt(args.output, tw_pred_ref, delimiter=",")