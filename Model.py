from concurrent.futures._base import Error
import cornac
import os
import numpy as np
from cornac.data import reader
from cornac.eval_methods import RatioSplit
from cornac import experiment
from cornac.data import Reader
from resource import *
from string import ascii_letters, punctuation, whitespace
import time
import csv
import re
import fcntl
from cornac.models.mf.recom_mf import MF
from cornac.models.ncf.recom_neumf import NeuMF


# Model class @Author Michael Miller 2205885M
# Utilises the Cornac framework [Salah et al 2020]
# Cornac's open source framework is accessible at https://github.com/PreferredAI/cornac.
# model class houses all memory/time/metric capturing functionality.
# Additionally, it houses all functionality needed for running experiments


# method strips vm stats from proc file specific to the active process id
# return lines[11:23], only those lines containing virtual memory statistics
def get_memory():
    pid = os.getpid()
    with open("/proc/%s/status" % pid, "r") as proc:
        lines = [line.strip() for line in proc]
        return lines[11:23]


# method uses regex to strip numeric values from a get_memory() call
# returns values, a list containing only those values
def mem_capture():
    values = []
    lines = get_memory()
    for line in lines:
        match = re.search(r'([0-9]+)', line)
        if match:
            values.append(int(match.group()))
    return values


# method calculates total memory consumption of a process through subtracting before from after
# @param parsed contents of virtual memory statistics
# calculate difference in memory consumption (after - before)
# return vm stats for printing
def memory_calculation(mem_contents):
    after_shot = mem_capture()
    vmpeak = after_shot[0] - mem_contents[0]
    vmsize = after_shot[1] - mem_contents[1]
    vmlck = after_shot[2] - mem_contents[2]
    vmpin = after_shot[3] - mem_contents[3]
    vmhwm = after_shot[4] - mem_contents[4]
    vmrss = after_shot[5] - mem_contents[5]
    vmdata = after_shot[6] - mem_contents[6]
    vmstk = after_shot[7] - mem_contents[7]
    vmexe = after_shot[8] - mem_contents[8]
    vmlib = after_shot[9] - mem_contents[9]
    vmpte = after_shot[10] - mem_contents[10]
    vmswap = after_shot[11] - mem_contents[11]
    return [vmpeak, vmsize, vmlck, vmpin, vmhwm, vmrss,
            vmdata, vmstk, vmexe, vmlib, vmpte, vmswap]


# method captures time data of active process
# from getruusage() only take firs two values, corresponding to system and user time
# return list of those values for analysis
def time_capture():
    list_time = getrusage(RUSAGE_SELF)
    ru_utime = list_time[0]
    ru_stime = list_time[1]
    return [ru_utime, ru_stime]


# method calculates total time taken by process by subtracting before from after
# @params system/user time data from before training/testing
# return after-before time for accurate measurements of training/testing time
def time_calculation(before_time):
    time_taken = time_capture()
    time_taken[0] = time_taken[0] - before_time[0]
    time_taken[1] = time_taken[1] - before_time[1]
    return time_taken


# method creates the Cornac metrics and models required to run experiments, then performs experiments.
# @param model: RecModel object created in main method housing relevant input parameter values
#        ratio_split: data set for use in experiment, split into 0.8/0.2 train/test split by Cornac's method
# Returns metrics, list containing the numeric values contained in the results of the experiment
def run_experiment(model, ratio_split):
    # metrics (Accuracy/rating)
    mae = cornac.metrics.MAE()  # mean absolute error
    mse = cornac.metrics.MSE()  # mean squared error
    rmse = cornac.metrics.RMSE()  # root mean squared error
    # metrics (Accuracy/Ranking)
    auc = cornac.metrics.AUC()  # area under curve
    f1 = cornac.metrics.FMeasure(k=-1)  # f measure (utility of item per user)
    rec_10 = cornac.metrics.Recall(k=10)  # recall
    pre_10 = cornac.metrics.Precision(k=10)  # precision
    ndcg = cornac.metrics.NDCG()  # normalised dcg = (discount cumulative gain/ideal discount cumulative gain)
    ncrr = cornac.metrics.NCRR()  # normalised cumulative reciprocal rank
    mAp = cornac.metrics.MAP()  # mean average precision
    mrr = cornac.metrics.MRR()  # mean reciprocal rank
    the_metrics = {"mae": mae, "mse": mse, "rmse": rmse, "auc": auc,
                   "fmeasure": f1, "rec10": rec_10, "pre10": pre_10,
                   "ndcg": ndcg, "ncrr": ncrr, "map": mAp, "mmr": mrr}
    if model.get_model_type() == 1:
        mf = cornac.models.MF(
            k=model.get_k(),  # 10
            max_iter=model.get_max_iter(),  # 25
            learning_rate=model.get_learning_rate(),  # 0.01
            lambda_reg=model.get_lambda_reg(),  # 0.02
            use_bias=model.get_use_bias(),  # True
            early_stop=model.get_early_stop(),  # True
            verbose=model.get_verbose(),  # True
        )
        # create experiment
        exp = cornac.Experiment(
            eval_method=ratio_split,
            models=[mf],
            metrics=[mae, mse, rmse, auc, f1, rec_10, pre_10, ndcg, ncrr, mAp, mrr],
            save_dir="trained.csv"
        )
        # run experiment
        exp.run()
        # convert results into list of strings
        lister = []
        lister = str(exp.result)
        # extract only the values of the metrics from the list
        metrics = lister[277:len(lister)]
        return metrics

    if model.get_model_type() == 2:
        bpr = cornac.models.BPR(
            k=model.get_k(),
            max_iter=model.get_max_iter(),
            learning_rate=model.get_learning_rate(),
            lambda_reg=model.get_lambda_reg(),
            verbose=model.get_verbose()
        )
        # create experiment
        exp = cornac.Experiment(
            eval_method=ratio_split,
            models=[bpr],
            metrics=[mae, mse, rmse, auc, f1, rec_10, pre_10, ndcg, ncrr, mAp, mrr],
            save_dir="trained.csv"
        )
        # run experiment
        exp.run()
        # convert results into list of strings
        lister = []
        lister = str(exp.result)
        # extract only the values of the metrics from the list
        metrics = lister[277:len(lister)]
        return metrics
    if model.get_model_type() == 3:
        neumf = cornac.models.NeuMF(
            num_factors=model.get_k(),
            layers=model.get_layers(),
            act_fn=model.get_act_fn(),
            learner=model.get_learner(),
            num_epochs=model.get_max_iter(),
            batch_size=model.get_batch_size(),
            lr=model.get_learning_rate(),
            num_neg=model.get_num_neg()
        )
        exp = cornac.Experiment(
            eval_method=ratio_split,
            models=[neumf],
            metrics=[mae, mse, rmse, auc, f1, rec_10, pre_10, ndcg, ncrr, mAp, mrr],
            save_dir="trained.csv"
        )
        # run experiment
        exp.run()
        # convert results into list of strings
        lister = []
        lister = str(exp.result)
        # extract only the values of the metrics from the list
        metrics = lister[277:len(lister)]
        return metrics


# method uses reg ex to strip only numeric values corresponding to each metric for use in file writing
# @param metrics: the list taken from experiments results
# use regex to extract float values of accuracy/rank metrics and add them to values list
# return values, list of values corresponding to each metric
def metric_capture(metrics):
    results = " "
    values = []
    for line in metrics:
        results += line
    metrics = results.split(" | ")
    for line in metrics:
        match = re.search(r'(\d*\.\d+|\d+\d+\d)+', str(line))
        if match:
            values.append(float(match.group()))
    return values


