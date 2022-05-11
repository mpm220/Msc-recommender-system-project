import gc
import random
import time
import cornac
from cornac.eval_methods import RatioSplit
import tensorflow as tf
import BPR
import Model
from cornac.data import Reader
from sklearn.utils import resample
import NCF
import Writer
import os
from multiprocessing import Process, current_process, Pool
import RandomSeed
import MF
import BootStraps

# Main class, @Author Michael Miller 2205885M
# class serves as the driver for the whole script, all coordination between classes is done within its methods.
# the class utilises many features of the Cornac framework [Salah et al 2020]
# Cornac's open source framework is accessible at https://github.com/PreferredAI/cornac.


# method coordinates model creation,
# predetermined model configurations are specified here to be supplied to Model for experiments
# return model_list: list containing all models for experiments
def model_maker():
    model_list = []
    # # --------------------------------------- MF models
    # create max iter models
    for x in range(25, 80, 25):
        model = MF.MF("max_iterations = %s" % x, 10, x, 0.01, 0.02, False, False, False)
        model_list.append(model)
    # create k models
    for x in range(10, 31, 10):
        model = MF.MF("k = %s" % x, x, 25, 0.01, 0.02, False, False, False)
        model_list.append(model)
    # create learning rate models
    learn_rate_floats = [0.01, 0.03, 0.06]
    for x in learn_rate_floats:
        model = MF.MF("learning_rate = %s" % x, 10, 25, x, 0.02, False, False, False)
        model_list.append(model)
    # create lambda reg models
    lambda_rate_floats = [0.02, 0.06, 0.09]
    for x in lambda_rate_floats:
        model = MF.MF("lambda_reg = %s" % x, 10, 25, 0.01, x, False, False, False)
        model_list.append(model)
    # use_bias models
    mf_bias_one = MF.MF("use_bias = True", 10, 25, 0.01, 0.02, True, False, False)
    mf_bias_two = MF.MF("use_bias = False", 10, 25, 0.01, 0.02, False, False, False)
    model_list.append(mf_bias_one)
    model_list.append(mf_bias_two)
    # early_stop models
    mf_early_stop_one = MF.MF("early_stop = True", 10, 25, 0.01, 0.02, False, True, False)
    mf_early_stop_two = MF.MF("early_stop = False", 10, 25, 0.01, 0.02, False, False, False)
    model_list.append(mf_early_stop_one)
    model_list.append(mf_early_stop_two)
    # # verbose models
    mf_verbose_one = MF.MF("verbose = True", 10, 25, 0.01, 0.02, False, False, True)
    mf_verbose_two = MF.MF("verbose = False", 10, 25, 0.01, 0.02, False, False, False)
    model_list.append(mf_verbose_one)
    model_list.append(mf_verbose_two)
    # --------------------------------------------- BPR models
    # mx iter
    for x in range(25, 80, 25):
        model = BPR.BPR("(BPR) max_iterations = %s" % x, 10, x, 0.01, 0.02, False)
        model_list.append(model)
    # k models
    for x in range(10, 31, 10):
        model = BPR.BPR("(BPR) k = %s" % x, x, 25, 0.01, 0.02, False)
        model_list.append(model)
    # learn rate models
    for x in learn_rate_floats:
        model = BPR.BPR("(BPR) learning_rate = %s" % x, 10, 25, x, 0.02, False)
        model_list.append(model)
    # lambda models
    for x in lambda_rate_floats:
        model = BPR.BPR("(BPR) lambda_reg = %s" % x, 10, 25, 0.01, x, False)
        model_list.append(model)
    # verbose models
    bpr_verb_T = BPR.BPR("(BPR) verbose = true", 10, 25, 0.01, 0.02, True)
    bpr_verb_F = BPR.BPR("(BPR) verbose = false", 10, 25, 0.01, 0.02, False)
    model_list.append(bpr_verb_T)
    model_list.append(bpr_verb_F)
    # # -------------------------------- NEUMF models
    layers_default = [64, 32, 16, 8]
    # num_factors models
    for x in range(8, 25, 8):
        model = NCF.NCF("(NeuralMF1) num_factors = %s" % x, x, layers_default, "tanh", "adam", 10, 256,
                        0.001, 50)
        model_list.append(model)
    layers1 = [128, 64, 32, 16]
    # layers models
    neumf_layer1 = NCF.NCF("(NeuralMF1) layers = 64:8", 8, layers_default, "tanh", "adam", 10, 256,
                           0.001, 50)
    neumf_layer2 = NCF.NCF("(NeuralMF1) layers = 128:16", 8, layers1, "tanh", "adam", 10, 256,
                           0.001, 50)
    model_list.append(neumf_layer1)
    model_list.append(neumf_layer2)
    # activation function models
    list_act_fn = ["tanh", "relu", "sigmoid"]
    for word in list_act_fn:
        model = NCF.NCF("(NeuralMF1) activation function = %s" % word, 8, layers_default, word, "adam", 10,
                        256, 0.001, 50)
        model_list.append(model)
    # learner models
    list_optimisers = ["adam", "rmsprop", "adagrad", "sdg"]
    for word in list_optimisers:
        model = NCF.NCF("(NeuralMF1) optimiser = %s" % word, 8, layers_default, "tanh", word, 10, 256,
                        0.001, 50)
        model_list.append(model)
    # num epochs models
    for x in range(10, 31, 10):
        model = NCF.NCF("(NeuralMF1) epoch count = %s" % x, 8, layers_default, "tanh", "adam", 10, 256,
                        0.001, 50)
        model_list.append(model)
    # batch size models
    batch_256 = NCF.NCF("(NeuralMF1) batch size = 256", 8, layers_default, "tanh", "adam", 10, 256,
                        0.001, 50)
    batch_512 = NCF.NCF("(NeuralMF1) batch size = 512", 8, layers_default, "tanh", "adam", 10, 512,
                        0.001, 50)
    model_list.append(batch_256)
    model_list.append(batch_512)
    # learn rate models
    learn_rates = [0.001, 0.002, 0.006]
    for x in learn_rates:
        model = NCF.NCF("(NeuralMF1) learn rate = %s" % x, 8, layers_default, "tanh", "adam", 10, 256,
                        x, 50)
        model_list.append(model)
    # num_neg models
    for x in range(50, 101, 50):
        model = NCF.NCF("(NeuralMF1) negative pairing = %s" % x, 8, layers_default, "tanh", "adam", 10, 256,
                        0.001, x)
        model_list.append(model)
    return model_list


# method coordinates BootStrap creation,
# 10-100 percent samples are specified here
# return model_list: list containing all samples
def bootstrap_maker():
    full = BootStraps.BootStraps(100, 100)
    ninety = BootStraps.BootStraps(90, 90)
    eighty = BootStraps.BootStraps(80, 88.8888)
    seventy = BootStraps.BootStraps(70, 87.5)
    sixty = BootStraps.BootStraps(60, 85.7142)
    fifty = BootStraps.BootStraps(50, 83.3333)
    forty = BootStraps.BootStraps(40, 80)
    thirty = BootStraps.BootStraps(30, 75)
    twenty = BootStraps.BootStraps(20, 66.6666)
    ten = BootStraps.BootStraps(10, 50)
    samples = [ten, twenty, thirty, forty, fifty, sixty, seventy, eighty, ninety, full]
    return samples


# helper method to supply random seed generation
# return random integer
def random_number():
    return random.randint(1, 300)


# method creates defined sized list of random integers
# return this list: seed_list
def seed_maker(x):
    seed_list = []
    for i in range(x):
        seed_list.append(RandomSeed.RandomSeed(random_number()))
    return seed_list


# method performs 0.8/0.2 train/test  ratio split of supplied dataset
# @param re_sample: randomly sampled subset of the movielens data,
# return ratio_splitter: a shuffled and split data-sets
def get_ratio_split(re_sample):
    ratio_splitter = cornac.eval_methods.ratio_split.RatioSplit(data=re_sample, test_size=0.2,
                                                                exclude_unknowns=False,
                                                                verbose=True)
    return ratio_splitter


# method generates sub-samples required for experiment
# @params x: required size of resampled data set
#         percentage: required percentage to obtain x sized resample
#         data_set: sample from which to resample
#         whole_data: full dataset from which each sub-sample is resampled
# returns boot_sample: resample of the required size
def sample_gen(x, percentage, data_set, whole_data):
    # obtain ten% of sample size and use to adjust each bootstrap objects sample
    # to take from the correct sized sample
    adjustment = int(len(whole_data) / 10)
    if x == 100 or x == 90:
        return data_set
    else:
        boot_sample = resample(data_set, replace=False,
                               n_samples=calculator(percentage, len(data_set) + adjustment))
        return boot_sample


# Helper method to create bootstrap samples
# @param          x: the percentage to be calculated
#        total_size: size of data sets
# return the number of data required to populate sample
def calculator(x, size):
    number = int(x / 100 * size)
    return number


# method to coordinate before/after memory and time writing via Writer.py
# @params x: integer to diferentiate before from after file to write to
#         bootstrap: bootstrap object to write to file
#         model: RecModel object to write to file
#         random_seed: RandomSeed object to write to file
def mem_time_before_after(x, bootstrap, model, random_seed):
    if x == 1:
        # time
        before = Model.time_capture()
        Writer.time_snapshot_before(before, bootstrap.get_name(), model.get_name(), random_seed.get_string())
        # mem
        one_mem = Model.mem_capture()
        Writer.mem_snapshot_before(one_mem, bootstrap.get_name(), model.get_name(), random_seed.get_string())
    else:
        # time
        after = Model.time_capture()
        Writer.time_snapshot_after(after, bootstrap.get_name(), model.get_name(), random_seed.get_string())
        # mem
        one_mem = Model.mem_capture()
        Writer.mem_snapshot_after(one_mem, bootstrap.get_name(), model.get_name(), random_seed.get_string())


# method encapsulates each process spawned in the main method in Main.py
# @params model: RecModel object to supply for experiment and file writing
#         bootstrap: BootStraps object to supply samples for experiment and file writing
#         random_seed: RandomSeed object to supply random seed for initial resample call
# takes before and after memory/time capture, sends them to Writer.py and Model.py to calculatie the difference
# sends results to Writer.py for file writing
def train_test(model, bootstrap, random_seed):
    print(os.getpid())
    gc.collect()
    cornac_split = get_ratio_split(bootstrap.get_sample())
    # write before snapshots
    mem_time_before_after(1, bootstrap, model, random_seed)
    # timestamp/ memory snapshot
    before_time = Model.time_capture()
    before_mem = Model.mem_capture()
    metrics = Model.run_experiment(model, cornac_split)
    # write after snapshots
    mem_time_before_after(0, bootstrap, model, random_seed)
    # TIME/MEMORY CALCULATIONS  plus metrics write
    Writer.metric_write(Model.metric_capture(metrics), bootstrap.get_name(), model.get_name(), random_seed.get_string())
    Writer.mem_write(Model.memory_calculation(before_mem), bootstrap.get_name(), model.get_name(),
                     random_seed.get_string())
    Writer.time_write(Model.time_calculation(before_time), bootstrap.get_name(), model.get_name(),
                      random_seed.get_string())
    gc.collect()


# If run as main, create list of processes
# use list of models and list of bootstrap samples to populate processes
# run processes and wait for each's conclusion before starting the next ones.
if __name__ == '__main__':
    # create models
    model_list = model_maker()
    # create bootstraps
    samples = bootstrap_maker()
    # create random seed objects
    random_seeds = seed_maker(5)
    # instantiate data set
    data_1M = cornac.datasets.movielens.load_feedback(variant="1M", reader=Reader())
    for y in model_list:
        for x in samples:
            process_list = []
            for z in random_seeds:
                # set datasample of bootstrap = to size ie 90 = 90%
                x.set_sample(resample(data_1M, replace=False, n_samples=calculator(x.get_size(), len(data_1M)),
                                      random_state=z.get_size()))
                # random state here increment throughout process
                # ignore if 100 or 90,
                # take 80% sample from 90% sample, adjust percentages to remain 80% sample of data set
                x.set_sample(sample_gen(x.get_size(), x.get_bootstrap_percentage(), x.get_sample(), data_1M))
                process = Process(target=train_test, args=(y, x, z))
                process.start()
                process_list.append(process)
            # once 5 random seeds per sample processes have begun wait on join, then close to free resource
            # for next batch
            for process in process_list:
                process.join()
            process.close()


