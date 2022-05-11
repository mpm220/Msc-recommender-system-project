from concurrent.futures._base import Error
import time
import csv
import pandas as pd
import re
import fcntl
from cornac.models.mf.recom_mf import MF
import os
import subprocess

# Writer class @Author Michael Miller 2205885M
# class houses all file writing functionality for the script, split into memory time and metrics sections


# -------------------------------MEMORY------------------------------------------
# @param mem_record: list of values corresponding to virtual memory data
#        name: string denoting sample size
#        model_param: string denoting changed model parameter
#        random_seed: string denoting random_seed value used in experiment
# method writes memory data to file
def mem_snapshot_before(mem_record, name, model_param, random_seed):
    # if file does not exist write header
    if not os.path.isfile('before_system_memory_copy.csv'):
        with open("before_system_memory_copy.csv", "w") as rec:
            fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
            writer = csv.writer(rec)
            writer.writerow(["model_param", "sample_size", "random_seed", "vmpeak(kb)",
                             "vmsize(kb)", "vmlck(kb)", "vmpin", "vmhwm(kb)", "vmrss(kb)",
                             "vmdata(kb)", "vmstk(kb)", "vmexe(kb)", "vmlib(kb)", "vmpte(kb)", "vmswap(kb)"])
            fcntl.flock(rec, fcntl.LOCK_UN)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            # insert key value columns for model identification
            mem_record.insert(0, random_seed)
            mem_record.insert(0, name)
            mem_record.insert(0, model_param)
            with open("before_system_memory_copy.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in mem_record])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
            break
        except Error:
            # if file locked wait 0.05 seconds and try again
            time.sleep(0.05)


# @param mem_record: list of values corresponding to virtual memory data
#        name: string denoting sample size
#        model_param: string denoting changed model parameter
#        random_seed: string denoting random_seed value used in experiment
# method writes memory data to file
def mem_snapshot_after(mem_record, name, model_param, random_seed):
    # if file does not exist write header
    if not os.path.isfile('after_system_memory_copy.csv'):
        with open("after_system_memory_copy.csv", "w") as rec:
            fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
            writer = csv.writer(rec)
            writer.writerow(["model_param", "sample_size", "random_seed", "vmpeak(kb)",
                             "vmsize(kb)", "vmlck(kb)", "vmpin", "vmhwm(kb)", "vmrss(kb)",
                             "vmdata(kb)", "vmstk(kb)", "vmexe(kb)", "vmlib(kb)", "vmpte(kb)", "vmswap(kb)"])
            fcntl.flock(rec, fcntl.LOCK_UN)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            # insert key value columns for model identification
            mem_record.insert(0, random_seed)
            mem_record.insert(0, name)
            mem_record.insert(0, model_param)
            with open("after_system_memory_copy.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in mem_record])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            # if file locked wait 0.05 seconds and try again
            time.sleep(0.05)


# @param mem_calc: list of values corresponding to virtual memory data
#        name: string denoting sample size
#        model_param: string denoting changed model parameter
#        random_seed: string denoting random_seed value used in experiment
# method writes memory data to file
def mem_write(mem_calc, name, model_param, random_seed):
    # if file does not exist write header
    if not os.path.isfile('Memory_Recordings.csv'):
        with open("Memory_Recordings.csv", "w") as rec:
            fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
            writer = csv.writer(rec)
            writer.writerow(["model_param", "sample_size", "random_seed", "vmpeak(kb)",
                             "vmsize(kb)", "vmlck(kb)", "vmpin", "vmhwm(kb)", "vmrss(kb)",
                             "vmdata(kb)", "vmstk(kb)", "vmexe(kb)", "vmlib(kb)", "vmpte(kb)", "vmswap(kb)"])
            fcntl.flock(rec, fcntl.LOCK_UN)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            # insert key value columns for model identification
            mem_calc.insert(0, random_seed)
            mem_calc.insert(0, name)
            mem_calc.insert(0, model_param)
            with open("Memory_Recordings.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in mem_calc])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            # if file locked wait 0.05 seconds and try again
            time.sleep(0.05)


# ----------------------------TIME------------------------------------------------
# @param time_read: list of float value corresponding to time capture data
#        name: string denoting sample size
#        model_param: string denoting changed model parameter
#        random_seed: string denoting random_seed value used in experiment
# method writes time data to file
def time_snapshot_before(time_read, name, model_param, random_seed):
    # if file does not exist write header
    if not os.path.isfile('before_time.csv'):
        with open("before_time.csv", "w") as rec:
            fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
            writer = csv.writer(rec)
            writer.writerow(["model_param", "sample_size", "random_seed", "user time", "system time"])
            fcntl.flock(rec, fcntl.LOCK_UN)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            # insert key value columns for model identification
            time_read.insert(0, random_seed)
            time_read.insert(0, name)
            time_read.insert(0, model_param)
            with open("before_time.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in time_read])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            # if file locked wait 0.05 seconds and try again
            time.sleep(0.05)


# @param time_read: list of float value corresponding to time capture data
#        name: string denoting sample size
#        model_param: string denoting changed model parameter
#        random_seed: string denoting random_seed value used in experiment
# method writes time data to file
def time_snapshot_after(time_read, name, model_param, random_seed):
    # if file does not exist write header
    if not os.path.isfile('after_time.csv'):
        with open("after_time.csv", "w") as rec:
            fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
            writer = csv.writer(rec)
            writer.writerow(["model_param", "sample_size", "random_seed", "user time", "system time"])
            fcntl.flock(rec, fcntl.LOCK_UN)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            # insert key value columns for model identification
            time_read.insert(0, random_seed)
            time_read.insert(0, name)
            time_read.insert(0, model_param)
            with open("after_time.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in time_read])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            # if file locked wait 0.05 seconds and try again
            time.sleep(0.05)


# @param time_calc: list of float value time recordings
#        name: string denoting sample size
#        model_param: string denoting changed model parameter
#        random_seed: string denoting random_seed value used in experiment
# method writes time data to file
def time_write(time_calc, name, model_param, random_seed):
    # if file does not exist write header
    if not os.path.isfile('Time_Recordings.csv'):
        with open("Time_Recordings.csv", "w") as rec:
            fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
            writer = csv.writer(rec)
            writer.writerow(["model_param", "sample_size", "random_seed", "user time", "system time"])
            fcntl.flock(rec, fcntl.LOCK_UN)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            # insert key value columns for model identification
            time_calc.insert(0, random_seed)
            time_calc.insert(0, name)
            time_calc.insert(0, model_param)
            with open("Time_Recordings.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in time_calc])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            time.sleep(0.05)  # wait before retrying


# ----------------------------- METRICS ---------------------------------------------
# @param metrics: list of float value metrics
#        name: string denoting sample size
#        model_param: string denoting changed model parameter
#        random_seed: string denoting random_seed value used in experiment
# method writes metric data to file
def metric_write(metrics, name, model_param, random_seed):
    # if file does not exist write header
    if not os.path.isfile('Metric_Recordings.csv'):
        with open("Metric_Recordings.csv", "w") as rec:
            fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
            writer = csv.writer(rec)
            writer.writerow(["model_param", "sample_size", "random_seed", "mae",
                             "mse", "rmse", "auc", "f1-1", "map", "mrr", "ncrr",
                             "ndcg", "precision", "recall", "train(s)", "test(s)"])
            fcntl.flock(rec, fcntl.LOCK_UN)
    # infinite loop to ensure each thread call results in data written to file
    while True:
        try:
            # insert key value columns for model identification
            metrics.insert(0, random_seed)
            metrics.insert(0, name)
            metrics.insert(0, model_param)
            with open("Metric_Recordings.csv", "a") as rec:
                # lock file
                fcntl.flock(rec, fcntl.LOCK_EX | fcntl.LOCK_NB)
                writer = csv.writer(rec)
                writer.writerow([r for r in metrics])
                # unlock file
                fcntl.flock(rec, fcntl.LOCK_UN)
                break
        except Error:
            time.sleep(0.05)
            # wait before retrying


