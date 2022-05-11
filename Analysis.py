import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.font_manager import FontProperties


# Analysis class, @Author Michael Miller 2205885M.
# class performs all data manipulation and visualisation functions required by this project.
# the class and its functions are separate from the rest of the script

# split apply combine method
# @param data: model/random seed data frame object
# method performs groupby and mean functions,
# for each sample/model combination, it calculates the mean value across all featured seeds.
# returns data_avg, the series object containing only those averages per sample/model combination
def split_apply_combine_mean(data):
    data_by_seeds = data.groupby(["sample_size"], sort=False)
    data_avg = data_by_seeds.mean()
    return data_avg


# method splits dataframe object into a list of smaller dataframes,
# each contains only the data relevant to that model/sample configuration
# @param data: the dataframe containing all results
# returns data_frame_list, list containing all divided sub-dataframes
def data_splitter(data):
    mf_max_iter_25 = data[0:50]
    mf_max_iter_50 = data[50:100]
    mf_max_iter_75 = data[100:150]
    mf_k_10 = data[150:200]
    mf_k_20 = data[200:250]
    mf_k_30 = data[250:300]
    mf_learn_rate01 = data[300:350]
    mf_learn_rate02 = data[350:400]
    mf_learn_rate06 = data[400:450]
    mf_lamb_reg02 = data[450:500]
    mf_lamb_reg06 = data[500:550]
    mf_lamb_reg09 = data[550:600]
    mf_use_bias_t = data[600:650]
    mf_use_bias_f = data[650:700]
    mf_early_stop_t = data[700:750]
    mf_early_stop_f = data[750:800]
    mf_verbose_t = data[800:850]
    mf_verbose_f = data[850:900]
    # BPR
    bpr_max_iter_25 = data[900:950]
    bpr_max_iter_50 = data[950:1000]
    bpr_max_iter_75 = data[1000:1050]
    bpr_k_10 = data[1050:1100]
    bpr_k_20 = data[1100:1150]
    bpr_k_30 = data[1150:1200]
    bpr_learn_rate01 = data[1200:1250]
    bpr_learn_rate03 = data[1250:1300]
    bpr_learn_rate06 = data[1300:1350]
    bpr_lambd_reg02 = data[1350:1400]
    bpr_lambd_reg06 = data[1400:1450]
    bpr_lambd_reg09 = data[1450:1500]
    bpr_verbose_t = data[1500:1550]
    bpr_verbose_f = data[1550:1600]
    # NEUMF
    nu_nfact_8 = data[1600:1650]
    nu_nfact_16 = data[1650:1700]
    nu_nfact_24 = data[1700:1750]
    nu_layers_8 = data[1750:1800]
    nu_layers_16 = data[1800:1850]
    nu_tanh = data[1850:1900]
    nu_relu = data[1900:1950]
    nu_sigmoid = data[1950:2000]
    nu_adam = data[2000:2050]
    nu_rmsprop = data[2050:2100]
    nu_adagrad = data[2100:2150]
    nu_sdg = data[2150:2200]
    nu_epoch10 = data[2200:2250]
    nu_epoch20 = data[2250:2300]
    nu_epoch30 = data[2300:2350]
    nu_batch256 = data[2350:2400]
    nu_batch512 = data[2400:2450]
    nu_lrate001 = data[2450:2500]
    nu_lrate002 = data[2500:2550]
    nu_lrate006 = data[2550:2600]
    nu_pair50 = data[2600:2650]
    nu_pair100 = data[2650:2700]
    # 1 MILLION
    # MF
    mf_mx25_1m = data[2700:2750]
    mf_mx50_1m = data[2750:2800]
    mf_mx75_1m = data[2800:2850]
    mf_k10_1m = data[2850:2900]
    mf_k20_1m = data[2900:2950]
    mf_k30_1m = data[2950:3000]
    mf_ler01_1m = data[3000:3050]
    mf_ler03_1m = data[3050:3100]
    mf_ler06_1m = data[3100:3150]
    mf_lam02_1m = data[3150:3200]
    mf_lam06_1m = data[3200:3250]
    mf_lam09_1m = data[3250:3300]
    mf_ubt_1m = data[3300:3350]
    mf_ubf_1m = data[3350:3400]
    mf_est_1m = data[3400:3450]
    mf_esf_1m = data[3450:3500]
    mf_vbt_1m = data[3500:3550]
    mf_vbf_1m = data[3550:3600]
    # BPR
    bpr_mx25_1m = data[3600:3650]
    bpr_mx50_1m = data[3650:3700]
    bpr_mx75_1m = data[3700:3750]
    bpr_k10_1m = data[3750:3800]
    bpr_k20_1m = data[3800:3850]
    bpr_k30_1m = data[3850:3900]
    bpr_lr01_1m = data[3900:3950]
    bpr_lr03_1m = data[3950:4000]
    bpr_lr06_1m = data[4000:4050]
    bpr_lam02_1m = data[4050:4100]
    bpr_lam06_1m = data[4100:4150]
    bpr_lam09_1m = data[4150:4200]
    bpr_vbt_1m = data[4200:4250]
    bpr_vbf_1m = data[4250:4300]
    data_frame_list = [mf_max_iter_25, mf_max_iter_50, mf_max_iter_75, mf_k_10, mf_k_20, mf_k_30,
                       mf_learn_rate01, mf_learn_rate02, mf_learn_rate06, mf_lamb_reg02, mf_lamb_reg06, mf_lamb_reg09,
                       mf_use_bias_t, mf_use_bias_f, mf_early_stop_t, mf_early_stop_f, mf_verbose_t, mf_verbose_f,
                       bpr_max_iter_25, bpr_max_iter_50, bpr_max_iter_75, bpr_k_10, bpr_k_20, bpr_k_30,
                       bpr_learn_rate01, bpr_learn_rate03, bpr_learn_rate06, bpr_lambd_reg02, bpr_lambd_reg06,
                       bpr_lambd_reg09, bpr_verbose_t, bpr_verbose_f, nu_nfact_8, nu_nfact_16, nu_nfact_24,
                       nu_layers_8, nu_layers_16, nu_tanh, nu_relu, nu_sigmoid, nu_adam, nu_rmsprop, nu_adagrad, nu_sdg,
                       nu_epoch10, nu_epoch20, nu_epoch30, nu_batch256, nu_batch512, nu_lrate001, nu_lrate002,
                       nu_lrate006, nu_pair50, nu_pair100, mf_mx25_1m, mf_mx50_1m, mf_mx75_1m, mf_k10_1m,
                       mf_k20_1m, mf_k30_1m, mf_ler01_1m, mf_ler03_1m, mf_ler06_1m, mf_lam02_1m, mf_lam06_1m,
                       mf_lam09_1m, mf_ubt_1m, mf_ubf_1m, mf_est_1m, mf_esf_1m, mf_vbt_1m, mf_vbf_1m, bpr_mx25_1m,
                       bpr_mx50_1m, bpr_mx75_1m, bpr_k10_1m, bpr_k20_1m, bpr_k30_1m, bpr_lr01_1m, bpr_lr03_1m,
                       bpr_lr06_1m, bpr_lam02_1m, bpr_lam06_1m, bpr_lam09_1m, bpr_vbt_1m, bpr_vbf_1m]
    return data_frame_list


# helper method, calculates magnitude of the error bars needed for each dataframe
# @param data: series object for analysis/graphing
# return error: magnitude of error bar
def error_calc(data):
    length = len(data)
    standard_dev = np.std(data)
    error = standard_dev / np.sqrt(length)
    return error


# method coordinates graphing all matrix factorisation data
# @param time_data: list of MF time data series objects
#        time_list: list of time data column names
#        metric_data: list of all MF metric data series objects
#        metric_list: list of metric data column names
#        metric_title_list: list of non-abbreviated metric names
#        memory_data: list of all MF memory data series objects
#        memory_stats: list of memory data column names
#        data_size: string denoting data size used in training/testing for those results
def mf_graph_maker(time_data, time_list, metric_data, metric_list, metric_title_list, memory_data, memory_stats,
                   data_size):
    data_title = "100k"
    if data_size == '1M':
        data_title = '1M'
    for x in range(0, len(time_list)):
        mf_times(time_data[0], time_data[1], time_data[2], time_data[3], time_data[4], time_data[5], time_data[6],
                 time_list[x], data_title)
    for x in range(0, len(metric_list)):
        mf_metrics(metric_data[0], metric_data[1], metric_data[2], metric_data[3], metric_data[4],
                   metric_data[5], metric_data[6], metric_list[x], metric_title_list[x], data_title)
    for x in range(0, len(memory_stats)):
        mf_memory(memory_data[0], memory_data[1], memory_data[2], memory_data[3], memory_data[4],
                  memory_data[5], memory_data[6], memory_stats[x], data_title)


# method plots Matrix factorisation model's memory data
# @param iter_memory: list containing iterations parameter model's data
#        k_memory: list containing k-models parameter model's  data
#        l_memory: list containing learning rate parameter model's data
#        lam_reg_memory: list containing lambda reg parameter model's data
#        ub_memory: list containing use bias parameter model's data
#        early_stop_memory: list containing early stop parameter model's data
#        verbose_memory: list containing verbose parameter model's data
#        memory_stats: string containing memory data column name
#        data_title: string denoting data_size used in training/testing for those results
def mf_memory(iter_memory, k_memory, l_memory, lam_reg_memory, ub_memory,
              early_stop_memory, verbose_memory, memory_stat, data_title):
    dir_name = data_title
    if str(data_title) == '100k':
        peak_range = [2300000, 2520000]
        data_range = [2300000, 2400000]
        hwn_range = [3000, 15000]
        pte_range = [250, 350]
        rss_range = [7000, 13000]
    else:
        peak_range = [2300000, 2520000]
        data_range = [2300000, 2400000]
        hwn_range = [0, 16000]
        pte_range = [200, 400]
        rss_range = [7000, 16000]
    if memory_stat == 'vmpeak(kb)':
        sub_plot_range = peak_range
    elif memory_stat == 'vmdata(kb)':
        sub_plot_range = data_range
    elif memory_stat == 'vmhwm(kb)':
        sub_plot_range = hwn_range
    elif memory_stat == 'vmpte(kb)':
        sub_plot_range = pte_range
    elif memory_stat == 'vmrss(kb)':
        sub_plot_range = rss_range
    else:
        sub_plot_range = [None, None]
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # --------------- ITER DATA
    mxt_25 = split_apply_combine_mean(iter_memory[0])
    m25 = mxt_25[memory_stat]
    mxt_50 = split_apply_combine_mean(iter_memory[1])
    m50 = mxt_50[memory_stat]
    mxt_75 = split_apply_combine_mean(iter_memory[2])
    m75 = mxt_75[memory_stat]
    # --------------- K MODELS DATA
    k10 = split_apply_combine_mean(k_memory[0])
    m10 = k10[memory_stat]
    k20 = split_apply_combine_mean(k_memory[1])
    m20 = k20[memory_stat]
    k30 = split_apply_combine_mean(k_memory[2])
    m30 = k30[memory_stat]
    # --------------- LEARN RATE DATA
    l01 = split_apply_combine_mean(l_memory[0])
    ml1 = l01[memory_stat]
    l03 = split_apply_combine_mean(l_memory[1])
    ml3 = l03[memory_stat]
    l06 = split_apply_combine_mean(l_memory[2])
    ml6 = l06[memory_stat]
    # -------------- L REG DATA
    reg02 = split_apply_combine_mean(lam_reg_memory[0])
    mr2 = reg02[memory_stat]
    reg06 = split_apply_combine_mean(lam_reg_memory[1])
    mr6 = reg06[memory_stat]
    reg09 = split_apply_combine_mean(lam_reg_memory[2])
    mr9 = reg09[memory_stat]
    # --------------- USE BIAS DATA
    ub_t = split_apply_combine_mean(ub_memory[0])
    m_ubt = ub_t[memory_stat]
    ub_f = split_apply_combine_mean(ub_memory[1])
    m_ubf = ub_f[memory_stat]
    # ---------------- EARLY STOP DATA
    es_t = split_apply_combine_mean(early_stop_memory[0])
    m_est = es_t[memory_stat]
    es_f = split_apply_combine_mean(early_stop_memory[1])
    m_esf = es_f[memory_stat]
    # ---------------- VERBOSE DATA
    vb_t = split_apply_combine_mean(verbose_memory[0])
    m_vbt = vb_t[memory_stat]
    vb_f = split_apply_combine_mean(verbose_memory[1])
    m_vbf = vb_f[memory_stat]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=4, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=1.1)
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    # ----------- ITER PLOT
    p1 = ax[0, 0].errorbar(p, m25, yerr=error_calc(m25), label="25")
    p2 = ax[0, 0].errorbar(p, m50, yerr=error_calc(m50), label="50")
    p3 = ax[0, 0].errorbar(p, m75, yerr=error_calc(m75), label="75")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel(memory_stat, fontsize='xx-small')
    ax[0, 0].set_title("max iteration", fontsize='x-small')
    ax[0, 0].legend(handles=[p1, p2, p3], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ K MODELS PLOT
    p4 = ax[0, 1].errorbar(p, m10, yerr=error_calc(m10), label="10")
    p5 = ax[0, 1].errorbar(p, m20, yerr=error_calc(m20), label="20")
    p6 = ax[0, 1].errorbar(p, m30, yerr=error_calc(m30), label="30")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel(memory_stat, fontsize='xx-small')
    ax[0, 1].set_title("K-models", fontsize='x-small')
    ax[0, 1].legend(handles=[p4, p5, p6], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ----------- LEARN RATE PLOT
    p7 = ax[1, 1].errorbar(p, ml1, yerr=error_calc(ml1), label="0.01")
    p8 = ax[1, 1].errorbar(p, ml3, yerr=error_calc(ml3), label="0.03")
    p9 = ax[1, 1].errorbar(p, ml6, yerr=error_calc(ml6), label="0.06")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel(memory_stat, fontsize='xx-small')
    ax[1, 1].set_title("learning rate", fontsize='x-small')
    ax[1, 1].legend(handles=[p4, p5, p6], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- LAM REG PLOT
    p10 = ax[1, 0].errorbar(p, mr2, yerr=error_calc(mr2), label="0.02")
    p11 = ax[1, 0].errorbar(p, mr6, yerr=error_calc(mr6), label="0.06")
    p12 = ax[1, 0].errorbar(p, mr9, yerr=error_calc(mr9), label="0.09")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel(memory_stat, fontsize='xx-small')
    ax[1, 0].set_title("Lambda reg", fontsize='x-small')
    ax[1, 0].legend(handles=[p10, p11, p12], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- USER BIAS PLOT
    p13 = ax[2, 0].errorbar(p, m_ubt, yerr=error_calc(m_ubt), label="true")
    p14 = ax[2, 0].errorbar(p, m_ubf, yerr=error_calc(m_ubf), label="false")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel(memory_stat, fontsize='xx-small')
    ax[2, 0].set_title("User bias", fontsize='x-small')
    ax[2, 0].legend(handles=[p13, p14], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- EARLY STOP PLOT
    p15 = ax[2, 1].errorbar(p, m_est, yerr=error_calc(m_est), label="true")
    p16 = ax[2, 1].errorbar(p, m_esf, yerr=error_calc(m_esf), label="false")
    ax[2, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 1].set_ylabel(memory_stat, fontsize='xx-small')
    ax[2, 1].set_title("Early stop", fontsize='x-small')
    ax[2, 1].legend(handles=[p15, p16], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ---------------- VERBOSE PLOT
    p17 = ax[3, 0].errorbar(p, m_vbt, yerr=error_calc(m_vbt), label="true")
    p18 = ax[3, 0].errorbar(p, m_vbf, yerr=error_calc(m_vbf), label="false")
    ax[3, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 0].set_ylabel(memory_stat, fontsize='xx-small')
    ax[3, 0].set_title("Verbose", fontsize='x-small')
    ax[3, 0].legend(handles=[p17, p18], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)

    ax[3, 1].axis('off')
    fig.suptitle('MF Model parameters and corresponding %s usage' % memory_stat)
    fig.savefig('Results/Graphs/MF/' + dir_name + '/Memory/%s.png' % memory_stat)
    plt.close(fig)


# method plots Matrix factorisation model's metrics data
# @param iter_metrics: list containing iterations parameter model's data
#        k_metrics: list containing k-models parameter model's  data
#        l_metrics: list containing learning rate parameter model's data
#        lam_reg_metrics: list containing lambda reg parameter model's data
#        ub_metrics: list containing use bias parameter model's data
#        early_stop_metrics: list containing early stop parameter model's data
#        verbose_metrics: list containing verbose parameter model's data
#        metric_name: string containing metric data column name
#        graph_title: string denoting metric name for use as title
#        data_title: string denoting data_size used in training/testing for those results
def mf_metrics(iter_metrics, k_metrics, l_metrics, lam_reg_metrics, ub_metrics, early_stop_metrics, verbose_metrics,
               metric_name, graph_title, data_title):
    dir_name = data_title
    sub_plot_range = []
    if str(data_title) == '100k':
        auc_range = [0.5, 0.7]
        f1_range = [0, 0.03]
        mae_range = [0.6, 1.2]
        map_range = [0.01, 0.05]
        mrr_range = [0, 0.17]
        mse_range = [None, None]
        ncrr_range = [0.02, 0.08]
        ndcg_range = [0.1, 0.4]
        prec_range = [0, 0.06]
        rec_range = [0.01, 0.04]
        rmse_range = [None, None]
    else:
        auc_range = [0.6, 0.9]
        f1_range = [0, 0.03]
        mae_range = [0.6, 1.2]
        map_range = [0, 0.05]
        mrr_range = [0, 0.2]
        mse_range = [None, None]
        ncrr_range = [0, 0.08]
        ndcg_range = [0, 0.4]
        prec_range = [0, 0.07]
        rec_range = [0, 0.03]
        rmse_range = [None, None]
    if metric_name == 'auc':
        sub_plot_range = auc_range
    elif metric_name == 'f1-1':
        sub_plot_range = f1_range
    elif metric_name == 'mae':
        sub_plot_range = mae_range
    elif metric_name == 'map':
        sub_plot_range = map_range
    elif metric_name == 'mrr':
        sub_plot_range = mrr_range
    elif metric_name == 'mse':
        sub_plot_range = mse_range
    elif metric_name == 'ncrr':
        sub_plot_range = ncrr_range
    elif metric_name == 'ndcg':
        sub_plot_range = ndcg_range
    elif metric_name == 'precision':
        sub_plot_range = prec_range
    elif metric_name == 'recall':
        sub_plot_range = rec_range
    elif metric_name == 'rmse':
        sub_plot_range = rmse_range
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # --------------- ITER DATA
    mxt_25 = split_apply_combine_mean(iter_metrics[0])
    mae25 = mxt_25[metric_name]
    mxt_50 = split_apply_combine_mean(iter_metrics[1])
    mae50 = mxt_50[metric_name]
    mxt_75 = split_apply_combine_mean(iter_metrics[2])
    mae75 = mxt_75[metric_name]
    # --------------- K MODELS DATA
    k10 = split_apply_combine_mean(k_metrics[0])
    mae10 = k10[metric_name]
    k20 = split_apply_combine_mean(k_metrics[1])
    mae20 = k20[metric_name]
    k30 = split_apply_combine_mean(k_metrics[2])
    mae30 = k30[metric_name]
    # --------------- LEARN RATE DATA
    l01 = split_apply_combine_mean(l_metrics[0])
    mael1 = l01[metric_name]
    l03 = split_apply_combine_mean(l_metrics[1])
    mael3 = l03[metric_name]
    l06 = split_apply_combine_mean(l_metrics[2])
    mael6 = l06[metric_name]
    # -------------- L REG DATA
    reg02 = split_apply_combine_mean(lam_reg_metrics[0])
    r2 = reg02[metric_name]
    reg06 = split_apply_combine_mean(lam_reg_metrics[1])
    r6 = reg06[metric_name]
    reg09 = split_apply_combine_mean(lam_reg_metrics[2])
    r9 = reg09[metric_name]
    # --------------- USE BIAS DATA
    ub_t = split_apply_combine_mean(ub_metrics[0])
    ubtmae = ub_t[metric_name]
    ub_f = split_apply_combine_mean(ub_metrics[1])
    ubfmae = ub_f[metric_name]
    # ---------------- EARLY STOP DATA
    es_t = split_apply_combine_mean(early_stop_metrics[0])
    estmae = es_t[metric_name]
    es_f = split_apply_combine_mean(early_stop_metrics[1])
    esfmae = es_f[metric_name]
    # ---------------- VERBOSE DATA
    vb_t = split_apply_combine_mean(verbose_metrics[0])
    vbt = vb_t[metric_name]
    vb_f = split_apply_combine_mean(verbose_metrics[1])
    vbf = vb_f[metric_name]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=4, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=1.1)
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    # ----------- ITER PLOT
    p1 = ax[0, 0].errorbar(p, mae25, yerr=error_calc(mae25), label="25")
    p2 = ax[0, 0].errorbar(p, mae50, yerr=error_calc(mae50), label="50")
    p3 = ax[0, 0].errorbar(p, mae75, yerr=error_calc(mae75), label="75")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 0].set_title("max iteration", fontsize='x-small')
    ax[0, 0].legend(handles=[p1, p2, p3], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ K MODELS PLOT
    p4 = ax[0, 1].errorbar(p, mae10, yerr=error_calc(mae10), label="10")
    p5 = ax[0, 1].errorbar(p, mae20, yerr=error_calc(mae20), label="20")
    p6 = ax[0, 1].errorbar(p, mae30, yerr=error_calc(mae30), label="30")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 1].set_title("K-models", fontsize='x-small')
    ax[0, 1].legend(handles=[p4, p5, p6], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ----------- LEARN RATE PLOT
    p7 = ax[1, 1].errorbar(p, mael1, yerr=error_calc(mael1), label="0.01")
    p8 = ax[1, 1].errorbar(p, mael3, yerr=error_calc(mael3), label="0.03")
    p9 = ax[1, 1].errorbar(p, mael6, yerr=error_calc(mael6), label="0.06")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 1].set_title("learning rate", fontsize='x-small')
    ax[1, 1].legend(handles=[p7, p8, p9], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- LAM REG PLOT
    p10 = ax[1, 0].errorbar(p, r2, yerr=error_calc(r2), label="0.02")
    p11 = ax[1, 0].errorbar(p, r6, yerr=error_calc(r6), label="0.06")
    p12 = ax[1, 0].errorbar(p, r9, yerr=error_calc(r9), label="0.09")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 0].set_title("Lambda reg", fontsize='x-small')
    ax[1, 0].legend(handles=[p10, p11, p12], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- USER BIAS PLOT
    p13 = ax[2, 0].errorbar(p, ubtmae, yerr=error_calc(ubtmae), label="true")
    p14 = ax[2, 0].errorbar(p, ubfmae, yerr=error_calc(ubfmae), label="false")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[2, 0].set_title("User bias", fontsize='x-small')
    ax[2, 0].legend(handles=[p13, p14], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- EARLY STOP PLOT
    p16 = ax[2, 1].errorbar(p, estmae, yerr=error_calc(estmae), label="true")
    p17 = ax[2, 1].errorbar(p, esfmae, yerr=error_calc(esfmae), label="false")
    ax[2, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[2, 1].set_title("Early stop", fontsize='x-small')
    ax[2, 1].legend(handles=[p16, p17], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ---------------- VERBOSE PLOT
    p18 = ax[3, 0].errorbar(p, vbt, yerr=error_calc(vbt), label="true")
    p19 = ax[3, 0].errorbar(p, vbf, yerr=error_calc(vbf), label="false")
    ax[3, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[3, 0].set_title("Verbose", fontsize='x-small')
    ax[3, 0].legend(handles=[p18, p19], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)

    ax[3, 1].axis('off')
    fig.suptitle('MF Model parameters and corresponding %s metrics' % graph_title)
    fig.savefig('Results/Graphs/MF/' + dir_name + '/Metrics/%s.png' % metric_name)
    fig.tight_layout()
    plt.close(fig)

# method plots Matrix factorisation model's time data
# @param use_bias_time: list containing use bias parameter model's data
#        early_stop_time: list containing early stop parameter model's data
#        verbose_time: list containing verbose parameter model's data
#        max_iter_time: list containing iterations parameter model's data
#        k_models_time: list containing k-models parameter model's  data
#        l_times: list containing learning rate parameter model's data
#        reg_times: list containing lambda reg parameter model's data
#        metric_name: string containing time data column name
#        data_title: string denoting data_size used in training/testing for those results
def mf_times(user_bias_time, early_stop_time, verbose_time, max_iter_time,
             k_models_time, l_times, reg_times, metric_name, data_title):
    dir_name = data_title
    sub_plot_range = []
    if str(data_title) == '100k':
        user_range = [35, 100]
        system_range = [5, 30]
    else:
        user_range = [400, 1000]
        system_range = [90, 210]
    if metric_name == 'user time':
        sub_plot_range = user_range
    elif metric_name == 'system time':
        sub_plot_range = system_range
    # --------------- TIME GRAPHS
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # ----------- USE BIAS
    ub_eq_t = split_apply_combine_mean(user_bias_time[0])
    ub_t_ut = ub_eq_t[metric_name]
    ub_eq_f = split_apply_combine_mean(user_bias_time[1])
    ub_f_ut = ub_eq_f[metric_name]
    # ----------- EARLY STOP
    es_eq_t = split_apply_combine_mean(early_stop_time[0])
    es_t_ut = es_eq_t[metric_name]
    es_eq_f = split_apply_combine_mean(early_stop_time[1])
    es_f_ut = es_eq_f[metric_name]
    # ----------- VERBOSE TIME
    vb_eq_t = split_apply_combine_mean(verbose_time[0])
    vb_t_ut = vb_eq_t[metric_name]
    vb_eq_f = split_apply_combine_mean(verbose_time[1])
    vb_f_ut = vb_eq_f[metric_name]
    # ------------ MAX ITER
    mxt_25 = split_apply_combine_mean(max_iter_time[0])
    mt_25_ut = mxt_25[metric_name]
    mxt_50 = split_apply_combine_mean(max_iter_time[1])
    mt_50_ut = mxt_50[metric_name]
    mxt_75 = split_apply_combine_mean(max_iter_time[2])
    mt_75_ut = mxt_75[metric_name]
    # ------------ K MODELS
    k10 = split_apply_combine_mean(k_models_time[0])
    k10_ut = k10[metric_name]
    k20 = split_apply_combine_mean(k_models_time[1])
    k20_ut = k20[metric_name]
    k30 = split_apply_combine_mean(k_models_time[2])
    k30_ut = k30[metric_name]
    # ------------ LEARNING RATE
    rate01 = split_apply_combine_mean(l_times[0])
    rate01_ut = rate01[metric_name]
    rate03 = split_apply_combine_mean(l_times[1])
    rate03_ut = rate03[metric_name]
    rate06 = split_apply_combine_mean(l_times[2])
    rate06_ut = rate06[metric_name]
    # -------------- LAMBDA REG
    reg02 = split_apply_combine_mean(reg_times[0])
    reg02_ut = reg02[metric_name]
    reg06 = split_apply_combine_mean(reg_times[1])
    reg06_ut = reg06[metric_name]
    reg09 = split_apply_combine_mean(reg_times[2])
    reg09_ut = reg09[metric_name]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=4, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=1.1)
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    # ------------ USE BIAS
    p1 = ax[0, 0].errorbar(p, ub_t_ut, yerr=error_calc(ub_t_ut), label="true")
    p2 = ax[0, 0].errorbar(p, ub_f_ut, yerr=error_calc(ub_f_ut), label="false")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[0, 0].set_title("user bias", fontsize='x-small')
    ax[0, 0].legend(handles=[p1, p2], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ EARLY STOP
    p3 = ax[0, 1].errorbar(p, es_t_ut, yerr=error_calc(es_t_ut), label="true")
    p4 = ax[0, 1].errorbar(p, es_f_ut, yerr=error_calc(es_f_ut), label="false")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[0, 1].set_title("Early Stop", fontsize='x-small')
    ax[0, 1].legend(handles=[p3, p4], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ VERBOSE TIME
    p5 = ax[1, 0].errorbar(p, vb_t_ut, yerr=error_calc(vb_t_ut), label="true")
    p6 = ax[1, 0].errorbar(p, vb_f_ut, yerr=error_calc(vb_f_ut), label="false")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[1, 0].set_title("Verbose", fontsize='x-small')
    ax[1, 0].legend(handles=[p5, p6], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ MAX ITER TIME
    p7 = ax[1, 1].errorbar(p, mt_25_ut, yerr=error_calc(mt_25_ut), label="25")
    p8 = ax[1, 1].errorbar(p, mt_50_ut, yerr=error_calc(mt_50_ut), label="50")
    p9 = ax[1, 1].errorbar(p, mt_75_ut, yerr=error_calc(mt_75_ut), label="75")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[1, 1].set_title("max iterations", fontsize='x-small')
    ax[1, 1].legend(handles=[p7, p8, p9], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- K MODELS PLOT
    p10 = ax[2, 0].errorbar(p, k10_ut, yerr=error_calc(k10_ut), label="10")
    p11 = ax[2, 0].errorbar(p, k20_ut, yerr=error_calc(k20_ut), label="20")
    p12 = ax[2, 0].errorbar(p, k30_ut, yerr=error_calc(k30_ut), label="30")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[2, 0].set_title("k models", fontsize='x-small')
    ax[2, 0].legend(handles=[p10, p11, p12], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LEARN RATE
    p13 = ax[2, 1].errorbar(p, rate01_ut, yerr=error_calc(rate01_ut), label="0.01")
    p14 = ax[2, 1].errorbar(p, rate03_ut, yerr=error_calc(rate03_ut), label="0.03")
    p15 = ax[2, 1].errorbar(p, rate06_ut, yerr=error_calc(rate06_ut), label="0.06")
    ax[2, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[2, 1].set_title("learning rate", fontsize='x-small')
    ax[2, 1].legend(handles=[p13, p14, p15], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- LAMBDA REG
    p16 = ax[3, 0].errorbar(p, reg02_ut, yerr=error_calc(reg02_ut), label="0.02")
    p17 = ax[3, 0].errorbar(p, reg06_ut, yerr=error_calc(reg06_ut), label="0.06")
    p18 = ax[3, 0].errorbar(p, reg09_ut, yerr=error_calc(reg09_ut), label="0.09")
    ax[3, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[3, 0].set_title("lambda reg", fontsize='x-small')
    ax[3, 0].legend(handles=[p16, p17, p18], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    fig.suptitle(" MF Model parameters and corresponding %s duration" % metric_name)
    ax[3, 1].axis('off')
    fig.savefig('Results/Graphs/MF/' + dir_name + '/Times/%s.png' % metric_name)
    plt.close(fig)


# method coordinates graphing all Bayes Personalised Ranking data
# @param time_data: list of BPR time data series objects
#        time_list: list of time data column names
#        memory_data: list of all BPR memory data series objects
#        memory_stats: list of memory data column names
#        metric_data: list of all BPR metric data series objects
#        metric_list: list of metric data column names
#        metric_title_list: list of non-abbreviated metric names
#        data_size: string denoting data size used in training/testing for those results
def bpr_graph_maker(time_data, time_list, memory_data, memory_stats, metric_data, metric_list, metric_titles,
                    data_size):
    data_title = "100k"
    if data_size == '1M':
        data_title = '1M'
    for x in range(0, len(time_list)):
        bpr_times(time_data[0], time_data[1], time_data[2], time_data[3], time_data[4], time_list[x],
                  data_title)
    for x in range(0, len(memory_stats)):
        bpr_memory(memory_data[0], memory_data[1], memory_data[2], memory_data[3], memory_data[4],
                   memory_stats[x], data_title)
    for x in range(0, len(metric_list)):
        bpr_metrics(metric_data[0], metric_data[1], metric_data[2], metric_data[3], metric_data[4],
                    metric_list[x], metric_titles[x], data_title)


# method plots BPR model's time data
# @param bpr_verbose_time: list containing verbose parameter model's data
#        bpr_iter_time: list containing iterations parameter model's data
#        bpr_k_times: list containing k-models parameter model's  data
#        bpr_l_time: list containing learning rate parameter model's data
#        bpr_lam_reg_time: list containing lambda reg parameter model's data
#        metric_name: string containing time data column name
#        data_title: string denoting data_size used in training/testing for those results
def bpr_times(bpr_iter_time, bpr_k_times, bpr_l_time, bpr_lam_reg_time, bpr_verbose_time, metric_name,
              data_title):
    dir_name = data_title
    sub_plot_range = []
    if str(data_title) == '100k':
        user_range = [40, 120]
        system_range = [5, 30]
    else:
        user_range = [400, 1000]
        system_range = [90, 210]
    if metric_name == 'user time':
        sub_plot_range = user_range
    elif metric_name == 'system time':
        sub_plot_range = system_range
    # --------------- TIME GRAPHS
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # ------------ MAX ITER
    mxt_25 = split_apply_combine_mean(bpr_iter_time[0])
    mt_25_ut = mxt_25[metric_name]
    mxt_50 = split_apply_combine_mean(bpr_iter_time[1])
    mt_50_ut = mxt_50[metric_name]
    mxt_75 = split_apply_combine_mean(bpr_iter_time[2])
    mt_75_ut = mxt_75[metric_name]
    # ------------ K MODELS
    k10 = split_apply_combine_mean(bpr_k_times[0])
    k10_ut = k10[metric_name]
    k20 = split_apply_combine_mean(bpr_k_times[1])
    k20_ut = k20[metric_name]
    k30 = split_apply_combine_mean(bpr_k_times[2])
    k30_ut = k30[metric_name]
    # ------------ LEARNING RATE
    rate01 = split_apply_combine_mean(bpr_l_time[0])
    rate01_ut = rate01[metric_name]
    rate03 = split_apply_combine_mean(bpr_l_time[1])
    rate03_ut = rate03[metric_name]
    rate06 = split_apply_combine_mean(bpr_l_time[2])
    rate06_ut = rate06[metric_name]
    # -------------- LAMBDA REG
    reg02 = split_apply_combine_mean(bpr_lam_reg_time[0])
    reg02_ut = reg02[metric_name]
    reg06 = split_apply_combine_mean(bpr_lam_reg_time[1])
    reg06_ut = reg06[metric_name]
    reg09 = split_apply_combine_mean(bpr_lam_reg_time[2])
    reg09_ut = reg09[metric_name]
    # ----------- VERBOSE TIME
    vb_eq_t = split_apply_combine_mean(bpr_verbose_time[0])
    vb_t_ut = vb_eq_t[metric_name]
    vb_eq_f = split_apply_combine_mean(bpr_verbose_time[1])
    vb_f_ut = vb_eq_f[metric_name]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.8)
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    # ------------ MAX ITER TIME
    p7 = ax[0, 0].errorbar(p, mt_25_ut, yerr=error_calc(mt_25_ut), label="25")
    p8 = ax[0, 0].errorbar(p, mt_50_ut, yerr=error_calc(mt_50_ut), label="50")
    p9 = ax[0, 0].errorbar(p, mt_75_ut, yerr=error_calc(mt_75_ut), label="75")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[0, 0].set_title("max iterations", fontsize='x-small')
    ax[0, 0].legend(handles=[p7, p8, p9], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- K MODELS PLOT
    p10 = ax[0, 1].errorbar(p, k10_ut, yerr=error_calc(k10_ut), label="10")
    p11 = ax[0, 1].errorbar(p, k20_ut, yerr=error_calc(k20_ut), label="20")
    p12 = ax[0, 1].errorbar(p, k30_ut, yerr=error_calc(k30_ut), label="30")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[0, 1].set_title("k models", fontsize='x-small')
    ax[0, 1].legend(handles=[p10, p11, p12], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LEARN RATE
    p13 = ax[1, 0].errorbar(p, rate01_ut, yerr=error_calc(rate01_ut), label="0.01")
    p14 = ax[1, 0].errorbar(p, rate03_ut, yerr=error_calc(rate03_ut), label="0.03")
    p15 = ax[1, 0].errorbar(p, rate06_ut, yerr=error_calc(rate06_ut), label="0.06")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[1, 0].set_title("learning rate", fontsize='x-small')
    ax[1, 0].legend(handles=[p13, p14, p15], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- LAMBDA REG
    p16 = ax[1, 1].errorbar(p, reg02_ut, yerr=error_calc(reg02_ut), label="0.02")
    p17 = ax[1, 1].errorbar(p, reg06_ut, yerr=error_calc(reg06_ut), label="0.06")
    p18 = ax[1, 1].errorbar(p, reg09_ut, yerr=error_calc(reg09_ut), label="0.09")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[1, 1].set_title("lambda reg", fontsize='x-small')
    ax[1, 1].legend(handles=[p16, p17, p18], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ VERBOSE TIME
    p5 = ax[2, 0].errorbar(p, vb_t_ut, yerr=error_calc(vb_t_ut), label="true")
    p6 = ax[2, 0].errorbar(p, vb_f_ut, yerr=error_calc(vb_f_ut), label="false")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[2, 0].set_title("Verbose", fontsize='x-small')
    ax[2, 0].legend(handles=[p5, p6], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    fig.suptitle(" BPR Model parameters and corresponding %s duration" % metric_name)
    ax[2, 1].axis('off')
    fig.savefig('Results/Graphs/BPR/' + dir_name + '/Times/%s.png' % metric_name)
    plt.close(fig)


# method plots BPR model's memory data
# @param bpr_verbose_memory: list containing verbose parameter model's data
#        bpr_iter_memory: list containing iterations parameter model's data
#        bpr_k_memory: list containing k-models parameter model's  data
#        bpr_l_memory: list containing learning rate parameter model's data
#        bpr_lam_reg_memory: list containing lambda reg parameter model's data
#        metric_name: string containing memory data column name
#        data_title: string denoting data_size used in training/testing for those results
def bpr_memory(bpr_iter_memory, bpr_k_memory, bpr_l_memory, bpr_lam_reg_memory,
               bpr_verbose_memory, metric_name, data_title):
    dir_name = data_title
    sub_plot_range = []
    if str(data_title) == '100k':
        peak_range = [2300000, 2520000]
        data_range = [2300000, 2400000]
        hwn_range = [3000, 15000]
        pte_range = [250, 350]
        rss_range = [7000, 13000]
    else:
        peak_range = [2300000, 2520000]
        data_range = [2300000, 2400000]
        hwn_range = [0, 16000]
        pte_range = [200, 400]
        rss_range = [7000, 16000]
    if metric_name == 'vmpeak(kb)':
        sub_plot_range = peak_range
    elif metric_name == 'vmdata(kb)':
        sub_plot_range = data_range
    elif metric_name == 'vmhwm(kb)':
        sub_plot_range = hwn_range
    elif metric_name == 'vmpte(kb)':
        sub_plot_range = pte_range
    elif metric_name == 'vmrss(kb)':
        sub_plot_range = rss_range
    else:
        sub_plot_range = [None, None]
    # --------------- MEMORY GRAPHS
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # ------------ MAX ITER
    mxt_25 = split_apply_combine_mean(bpr_iter_memory[0])
    mt_25_ut = mxt_25[metric_name]
    mxt_50 = split_apply_combine_mean(bpr_iter_memory[1])
    mt_50_ut = mxt_50[metric_name]
    mxt_75 = split_apply_combine_mean(bpr_iter_memory[2])
    mt_75_ut = mxt_75[metric_name]
    # ------------ K MODELS
    k10 = split_apply_combine_mean(bpr_k_memory[0])
    k10_ut = k10[metric_name]
    k20 = split_apply_combine_mean(bpr_k_memory[1])
    k20_ut = k20[metric_name]
    k30 = split_apply_combine_mean(bpr_k_memory[2])
    k30_ut = k30[metric_name]
    # ------------ LEARNING RATE
    rate01 = split_apply_combine_mean(bpr_l_memory[0])
    rate01_ut = rate01[metric_name]
    rate03 = split_apply_combine_mean(bpr_l_memory[1])
    rate03_ut = rate03[metric_name]
    rate06 = split_apply_combine_mean(bpr_l_memory[2])
    rate06_ut = rate06[metric_name]
    # -------------- LAMBDA REG
    reg02 = split_apply_combine_mean(bpr_lam_reg_memory[0])
    reg02_ut = reg02[metric_name]
    reg06 = split_apply_combine_mean(bpr_lam_reg_memory[1])
    reg06_ut = reg06[metric_name]
    reg09 = split_apply_combine_mean(bpr_lam_reg_memory[2])
    reg09_ut = reg09[metric_name]
    # ----------- VERBOSE TIME
    vb_eq_t = split_apply_combine_mean(bpr_verbose_memory[0])
    vb_t_ut = vb_eq_t[metric_name]
    vb_eq_f = split_apply_combine_mean(bpr_verbose_memory[1])
    vb_f_ut = vb_eq_f[metric_name]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.8)
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    # ------------ MAX ITER TIME
    p7 = ax[0, 0].errorbar(p, mt_25_ut, yerr=error_calc(mt_25_ut), label="25")
    p8 = ax[0, 0].errorbar(p, mt_50_ut, yerr=error_calc(mt_50_ut), label="50")
    p9 = ax[0, 0].errorbar(p, mt_75_ut, yerr=error_calc(mt_75_ut), label="75")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 0].set_title("max iterations", fontsize='x-small')
    ax[0, 0].legend(handles=[p7, p8, p9], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- K MODELS PLOT
    p10 = ax[0, 1].errorbar(p, k10_ut, yerr=error_calc(k10_ut), label="10")
    p11 = ax[0, 1].errorbar(p, k20_ut, yerr=error_calc(k20_ut), label="20")
    p12 = ax[0, 1].errorbar(p, k30_ut, yerr=error_calc(k30_ut), label="30")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 1].set_title("k models", fontsize='x-small')
    ax[0, 1].legend(handles=[p10, p11, p12], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LEARN RATE
    p13 = ax[1, 0].errorbar(p, rate01_ut, yerr=error_calc(rate01_ut), label="0.01")
    p14 = ax[1, 0].errorbar(p, rate03_ut, yerr=error_calc(rate03_ut), label="0.03")
    p15 = ax[1, 0].errorbar(p, rate06_ut, yerr=error_calc(rate06_ut), label="0.06")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 0].set_title("learning rate", fontsize='x-small')
    ax[1, 0].legend(handles=[p13, p14, p15], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- LAMBDA REG
    p16 = ax[1, 1].errorbar(p, reg02_ut, yerr=error_calc(reg02_ut), label="0.02")
    p17 = ax[1, 1].errorbar(p, reg06_ut, yerr=error_calc(reg06_ut), label="0.06")
    p18 = ax[1, 1].errorbar(p, reg09_ut, yerr=error_calc(reg09_ut), label="0.09")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 1].set_title("lambda reg", fontsize='x-small')
    ax[1, 1].legend(handles=[p16, p17, p18], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ VERBOSE TIME
    p5 = ax[2, 0].errorbar(p, vb_t_ut, yerr=error_calc(vb_t_ut), label="true")
    p6 = ax[2, 0].errorbar(p, vb_f_ut, yerr=error_calc(vb_f_ut), label="false")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[2, 0].set_title("Verbose", fontsize='x-small')
    ax[2, 0].legend(handles=[p5, p6], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    fig.suptitle(" BPR Model parameters and corresponding %s duration" % metric_name)
    ax[2, 1].axis('off')
    fig.savefig('Results/Graphs/BPR/' + dir_name + '/Memory/%s.png' % metric_name)
    plt.close(fig)


# method plots BPR model's metric data
# @param bpr_verbose_metrics: list containing verbose parameter model's data
#        bpr_iter_metrics: list containing iterations parameter model's data
#        bpr_k_metrics: list containing k-models parameter model's  data
#        bpr_l_metrics: list containing learning rate parameter model's data
#        bpr_lam_reg_metrics: list containing lambda reg parameter model's data
#        metric_name: string containing memory data column name
#        graph_title: string denoting metric name for use in graph title
#        data_title: string denoting data_size used in training/testing for those results
def bpr_metrics(bpr_iter_metrics, bpr_k_metrics, bpr_l_metrics, bpr_lam_reg_metrics,
                bpr_verbose_metrics, metric_name, graph_title, data_title):
    dir_name = data_title
    sub_plot_range = []
    if str(data_title) == '100k':
        auc_range = [0.7, 1]
        f1_range = [0, 0.03]
        mae_range = [1.7, 2.7]
        map_range = [0.0, 0.15]
        mrr_range = [0, 0.5]
        mse_range = [4, 9]
        ncrr_range = [0, 0.25]
        ndcg_range = [0.1, 0.6]
        prec_range = [0, 0.17]
        rec_range = [0.04, 0.13]
        rmse_range = [1.9, 2.9]
    else:
        auc_range = [0.7, 1]
        f1_range = [0, 0.03]
        mae_range = [1.7, 2.7]
        map_range = [0, 0.15]
        mrr_range = [0, 0.4]
        mse_range = [4, 9]
        ncrr_range = [0, 0.25]
        ndcg_range = [0.1, 0.6]
        prec_range = [0, 0.17]
        rec_range = [0.04, 0.13]
        rmse_range = [1.9, 2.9]
    if metric_name == 'auc':
        sub_plot_range = auc_range
    elif metric_name == 'f1-1':
        sub_plot_range = f1_range
    elif metric_name == 'mae':
        sub_plot_range = mae_range
    elif metric_name == 'map':
        sub_plot_range = map_range
    elif metric_name == 'mrr':
        sub_plot_range = mrr_range
    elif metric_name == 'mse':
        sub_plot_range = mse_range
    elif metric_name == 'ncrr':
        sub_plot_range = ncrr_range
    elif metric_name == 'ndcg':
        sub_plot_range = ndcg_range
    elif metric_name == 'precision':
        sub_plot_range = prec_range
    elif metric_name == 'recall':
        sub_plot_range = rec_range
    elif metric_name == 'rmse':
        sub_plot_range = rmse_range
    # --------------- METRIC GRAPHS
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # ------------ MAX ITER
    mxt_25 = split_apply_combine_mean(bpr_iter_metrics[0])
    mt_25_ut = mxt_25[metric_name]
    mxt_50 = split_apply_combine_mean(bpr_iter_metrics[1])
    mt_50_ut = mxt_50[metric_name]
    mxt_75 = split_apply_combine_mean(bpr_iter_metrics[2])
    mt_75_ut = mxt_75[metric_name]
    # ------------ K MODELS
    k10 = split_apply_combine_mean(bpr_k_metrics[0])
    k10_ut = k10[metric_name]
    k20 = split_apply_combine_mean(bpr_k_metrics[1])
    k20_ut = k20[metric_name]
    k30 = split_apply_combine_mean(bpr_k_metrics[2])
    k30_ut = k30[metric_name]
    # ------------ LEARNING RATE
    rate01 = split_apply_combine_mean(bpr_l_metrics[0])
    rate01_ut = rate01[metric_name]
    rate03 = split_apply_combine_mean(bpr_l_metrics[1])
    rate03_ut = rate03[metric_name]
    rate06 = split_apply_combine_mean(bpr_l_metrics[2])
    rate06_ut = rate06[metric_name]
    # -------------- LAMBDA REG
    reg02 = split_apply_combine_mean(bpr_lam_reg_metrics[0])
    reg02_ut = reg02[metric_name]
    reg06 = split_apply_combine_mean(bpr_lam_reg_metrics[1])
    reg06_ut = reg06[metric_name]
    reg09 = split_apply_combine_mean(bpr_lam_reg_metrics[2])
    reg09_ut = reg09[metric_name]
    # ----------- VERBOSE TIME
    vb_eq_t = split_apply_combine_mean(bpr_verbose_metrics[0])
    vb_t_ut = vb_eq_t[metric_name]
    vb_eq_f = split_apply_combine_mean(bpr_verbose_metrics[1])
    vb_f_ut = vb_eq_f[metric_name]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.7, hspace=0.8)
    plt.rcParams['ytick.labelsize'] = 'medium'
    plt.rcParams['xtick.labelsize'] = 'medium'
    # ------------ MAX ITER TIME
    p7 = ax[0, 0].errorbar(p, mt_25_ut, yerr=error_calc(mt_25_ut), label="25")
    p8 = ax[0, 0].errorbar(p, mt_50_ut, yerr=error_calc(mt_50_ut), label="50")
    p9 = ax[0, 0].errorbar(p, mt_75_ut, yerr=error_calc(mt_75_ut), label="75")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 0].set_title("max iterations", fontsize='x-small')
    ax[0, 0].legend(handles=[p7, p8, p9], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- K MODELS PLOT
    p10 = ax[0, 1].errorbar(p, k10_ut, yerr=error_calc(k10_ut), label="10")
    p11 = ax[0, 1].errorbar(p, k20_ut, yerr=error_calc(k20_ut), label="20")
    p12 = ax[0, 1].errorbar(p, k30_ut, yerr=error_calc(k30_ut), label="30")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 1].set_title("k models", fontsize='x-small')
    ax[0, 1].legend(handles=[p10, p11, p12], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LEARN RATE
    p13 = ax[1, 0].errorbar(p, rate01_ut, yerr=error_calc(rate01_ut), label="0.01")
    p14 = ax[1, 0].errorbar(p, rate03_ut, yerr=error_calc(rate03_ut), label="0.03")
    p15 = ax[1, 0].errorbar(p, rate06_ut, yerr=error_calc(rate06_ut), label="0.06")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 0].set_title("learning rate", fontsize='x-small')
    ax[1, 0].legend(handles=[p13, p14, p15], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- LAMBDA REG
    p16 = ax[1, 1].errorbar(p, reg02_ut, yerr=error_calc(reg02_ut), label="0.02")
    p17 = ax[1, 1].errorbar(p, reg06_ut, yerr=error_calc(reg06_ut), label="0.06")
    p18 = ax[1, 1].errorbar(p, reg09_ut, yerr=error_calc(reg09_ut), label="0.09")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 1].set_title("lambda reg", fontsize='x-small')
    ax[1, 1].legend(handles=[p16, p17, p18], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ VERBOSE TIME
    p5 = ax[2, 0].errorbar(p, vb_t_ut, yerr=error_calc(vb_t_ut), label="true")
    p6 = ax[2, 0].errorbar(p, vb_f_ut, yerr=error_calc(vb_f_ut), label="false")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[2, 0].set_title("Verbose", fontsize='x-small')
    ax[2, 0].legend(handles=[p5, p6], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    fig.suptitle('BPR Model parameters and corresponding %s metrics' % graph_title)
    ax[2, 1].axis('off')
    fig.savefig('Results/Graphs/BPR/' + dir_name + '/Metrics/%s.png' % metric_name)
    plt.close(fig)


# method coordinates graphing all Neural Collaborative filtering data
# @param time_data: list of BPR time data series objects
#        time_list: list of time data column names
#        memory_data: list of all BPR memory data series objects
#        memory_stats: list of memory data column names
#        metric_data: list of all BPR metric data series objects
#        metric_list: list of metric data column names
#        metric_titles: list of non-abbreviated metric names
#        data_size: string denoting data size used in training/testing for those results
def nu_graph_maker(time_data, time_list, memory_data, memory_stats, metric_data, metric_list, metric_titles,
                   data_size):
    data_title = "100k"
    if data_size == '1M':
        data_title = '1M'
    for x in range(0, len(time_list)):
        nu_time(time_data[0], time_data[1], time_data[2], time_data[3], time_data[4],
                time_data[5], time_data[6], time_data[7], time_list[x], data_title)
    for x in range(0, len(memory_stats)):
        nu_memory(memory_data[0], memory_data[1], memory_data[2], memory_data[3], memory_data[4],
                  memory_data[5], memory_data[6], memory_data[7], memory_stats[x], data_title)
    for x in range(0, len(metric_list)):
        nu_metrics(metric_data[0], metric_data[1], metric_data[2], metric_data[3], metric_data[4],
                   metric_data[5], metric_data[6], metric_data[7], metric_list[x], metric_titles[x],
                   data_title)


# method graphs NCF time data
# @param nu_factors_time: list containing num factors parameter model's data
#        nu_layers_time: list containing layers parameter model's data
#        nu_actf_time: list containing activation function parameter models's data
#        nu_opt_time: list containing optimisation function parameter model's data
#        nu_epoch_time: list containing epoch parameter model's data
#        nu_batch_time: list containing batch size parameter model's data
#        nu_lrate_time: list containing learn rate parameter model's data
#        nu_neg_time: list containing negative pairs parameter model's data
#        metric_name: string representing time data column names
#        data_title: string denoting data size used in training/testing of the model
def nu_time(nu_factors_time, nu_layers_time, nu_actf_time, nu_opt_time,
            nu_epoch_time, nu_batch_time, nu_lrate_time, nu_neg_time, metric_name, data_title):
    dir_name = data_title
    sub_plot_range = []
    if str(data_title) == '100k':
        user_range = [0, 2600]
        system_range = [0, 110]
    else:
        user_range = [400, 1000]
        system_range = [90, 210]
    if metric_name == 'user time':
        sub_plot_range = user_range
    elif metric_name == 'system time':
        sub_plot_range = system_range
    # --------------- TIME GRAPHS
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # ----------- NUM FACT
    nu_fact_8 = split_apply_combine_mean(nu_factors_time[0])
    n8 = nu_fact_8[metric_name]
    nu_fact_16 = split_apply_combine_mean(nu_factors_time[1])
    n16 = nu_fact_16[metric_name]
    nu_fact_24 = split_apply_combine_mean(nu_factors_time[2])
    n24 = nu_fact_24[metric_name]
    # ----------- LAYERS
    nu_layers8 = split_apply_combine_mean(nu_layers_time[0])
    layers1 = nu_layers8[metric_name]
    nu_layers16 = split_apply_combine_mean(nu_layers_time[1])
    layers2 = nu_layers16[metric_name]
    # ----------- ACTFN
    act1 = split_apply_combine_mean(nu_actf_time[0])
    tanh = act1[metric_name]
    act2 = split_apply_combine_mean(nu_actf_time[1])
    relu = act2[metric_name]
    act3 = split_apply_combine_mean(nu_actf_time[2])
    sigmoid = act3[metric_name]
    # ------------ OPT FUN
    opt1 = split_apply_combine_mean(nu_opt_time[0])
    adam = opt1[metric_name]
    opt2 = split_apply_combine_mean(nu_opt_time[1])
    rmsprop = opt2[metric_name]
    opt3 = split_apply_combine_mean(nu_opt_time[2])
    adagrad = opt3[metric_name]
    opt4 = split_apply_combine_mean(nu_opt_time[3])
    sdg = opt4[metric_name]
    # ------------ EPOCH
    E1 = split_apply_combine_mean(nu_epoch_time[0])
    E10 = E1[metric_name]
    E2 = split_apply_combine_mean(nu_epoch_time[1])
    E20 = E2[metric_name]
    E3 = split_apply_combine_mean(nu_epoch_time[2])
    E30 = E3[metric_name]
    # ------------ LEARNING RATE
    rate01 = split_apply_combine_mean(nu_lrate_time[0])
    rate01_ut = rate01[metric_name]
    rate03 = split_apply_combine_mean(nu_lrate_time[1])
    rate03_ut = rate03[metric_name]
    rate06 = split_apply_combine_mean(nu_lrate_time[2])
    rate06_ut = rate06[metric_name]
    # -------------- BATCH SIZE
    batch1 = split_apply_combine_mean(nu_batch_time[0])
    b252 = batch1[metric_name]
    batch2 = split_apply_combine_mean(nu_batch_time[1])
    b504 = batch2[metric_name]
    # -------------- NEG PAIRS
    neg1 = split_apply_combine_mean(nu_neg_time[0])
    neg50 = neg1[metric_name]
    neg2 = split_apply_combine_mean(nu_neg_time[1])
    neg50 = neg2[metric_name]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=4, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=1.3)
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    # ------------ FACTORS
    p1 = ax[0, 0].errorbar(p, n8, yerr=error_calc(n8), label="8")
    p2 = ax[0, 0].errorbar(p, n16, yerr=error_calc(n16), label="16")
    p90 = ax[0, 0].errorbar(p, n24, yerr=error_calc(n24), label="24")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[0, 0].set_title("Num Factors", fontsize='x-small')
    ax[0, 0].legend(handles=[p1, p2, p90], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LAYERS
    p3 = ax[0, 1].errorbar(p, layers1, yerr=error_calc(layers1), label="64:8")
    p4 = ax[0, 1].errorbar(p, layers2, yerr=error_calc(layers2), label="128:16")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[0, 1].set_title("Layers", fontsize='x-small')
    ax[0, 1].legend(handles=[p3, p4], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ ACTFN
    p5 = ax[1, 0].errorbar(p, tanh, yerr=error_calc(tanh), label="tanh")
    p6 = ax[1, 0].errorbar(p, relu, yerr=error_calc(relu), label="relu")
    p7 = ax[1, 0].errorbar(p, sigmoid, yerr=error_calc(sigmoid), label="sig")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[1, 0].set_title("Activation Function", fontsize='x-small')
    ax[1, 0].legend(handles=[p5, p6, p7], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ OPT FUN
    p8 = ax[1, 1].errorbar(p, adam, yerr=error_calc(adam), label="adam")
    p9 = ax[1, 1].errorbar(p, rmsprop, yerr=error_calc(rmsprop), label="rmsprop")
    p10 = ax[1, 1].errorbar(p, adagrad, yerr=error_calc(adagrad), label="adagrad")
    p11 = ax[1, 1].errorbar(p, sdg, yerr=error_calc(sdg), label="sdg")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[1, 1].set_title("Optimisation Function", fontsize='x-small')
    ax[1, 1].legend(handles=[p8, p9, p10, p11], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- EPOCH
    p12 = ax[2, 0].errorbar(p, E10, yerr=error_calc(E10), label="10")
    p13 = ax[2, 0].errorbar(p, E20, yerr=error_calc(E20), label="20")
    p14 = ax[2, 0].errorbar(p, E30, yerr=error_calc(E30), label="30")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[2, 0].set_title("Epoch count", fontsize='x-small')
    ax[2, 0].legend(handles=[p12, p13, p14], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LEARN RATE
    p15 = ax[2, 1].errorbar(p, rate01_ut, yerr=error_calc(rate01_ut), label="0.001")
    p16 = ax[2, 1].errorbar(p, rate03_ut, yerr=error_calc(rate03_ut), label="0.002")
    p17 = ax[2, 1].errorbar(p, rate06_ut, yerr=error_calc(rate06_ut), label="0.006")
    ax[2, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[2, 1].set_title("learning rate", fontsize='x-small')
    ax[2, 1].legend(handles=[p15, p16, p17], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- BATCH SIZE
    p18 = ax[3, 0].errorbar(p, b252, yerr=error_calc(b252), label="256")
    p19 = ax[3, 0].errorbar(p, b504, yerr=error_calc(b504), label="512")
    ax[3, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 0].set_ylabel("time (s)", fontsize='xx-small')
    ax[3, 0].set_title("Batch Size", fontsize='x-small')
    ax[3, 0].legend(handles=[p18, p19], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- NEG PAIRS
    p18 = ax[3, 1].errorbar(p, b252, yerr=error_calc(b252), label="50")
    p19 = ax[3, 1].errorbar(p, b504, yerr=error_calc(b504), label="100")
    ax[3, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 1].set_ylabel("time (s)", fontsize='xx-small')
    ax[3, 1].set_title("Negative pairing", fontsize='x-small')
    ax[3, 1].legend(handles=[p18, p19], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    fig.suptitle(" Neural MF Model parameters and corresponding %s duration" % metric_name)
    fig.savefig('Results/Graphs/Neumf/' + dir_name + '/Times/%s.png' % metric_name)
    plt.close(fig)


# method graphs NCF memory data
# @param nu_factors_memory: list containing num factors parameter model's data
#        nu_layers_memory: list containing layers parameter model's data
#        nu_actf_memory: list containing activation function parameter models's data
#        nu_opt_memory: list containing optimisation function parameter model's data
#        nu_epoch_memory: list containing epoch parameter model's data
#        nu_batch_memory: list containing batch size parameter model's data
#        nu_lrate_memory: list containing learn rate parameter model's data
#        nu_neg_memory: list containing negative pairs parameter model's data
#        metric_name: string representing memory data column names
#        data_title: string denoting data size used in training/testing of the model
def nu_memory(nu_factors_memory, nu_layers_memory, nu_actf_memory, nu_opt_memory,
              nu_epoch_memory, nu_batch_memory, nu_lrate_memory, nu_neg_memory, metric_name, data_title):
    dir_name = data_title
    if str(data_title) == '100k':
        peak_range = [7600000, 7800000]
        data_range = [7500000, 7650000]
        hwn_range = [None, None]
        pte_range = [None, None]
        rss_range = [110000, 145000]
        size_range = [7600000, 7800000]
    else:
        peak_range = [7600000, 7780000]
        data_range = [7500000, 7650000]
        hwn_range = [None, None]
        pte_range = [None, None]
        rss_range = [110000, 145000]
        size_range = [7640000, 7800000]
    if metric_name == 'vmpeak(kb)':
        sub_plot_range = peak_range
    elif metric_name == 'vmdata(kb)':
        sub_plot_range = data_range
    elif metric_name == 'vmhwm(kb)':
        sub_plot_range = hwn_range
    elif metric_name == 'vmpte(kb)':
        sub_plot_range = pte_range
    elif metric_name == 'vmrss(kb)':
        sub_plot_range = rss_range
    elif metric_name == 'vmsize(kb)':
        sub_plot_range = size_range
    else:
        sub_plot_range = [None, None]
    # --------------- TIME GRAPHS
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # ----------- NUM FACT
    nu_fact_8 = split_apply_combine_mean(nu_factors_memory[0])
    n8 = nu_fact_8[metric_name]
    nu_fact_16 = split_apply_combine_mean(nu_factors_memory[1])
    n16 = nu_fact_16[metric_name]
    nu_fact_24 = split_apply_combine_mean(nu_factors_memory[2])
    n24 = nu_fact_24[metric_name]
    # ----------- LAYERS
    nu_layers8 = split_apply_combine_mean(nu_layers_memory[0])
    layers1 = nu_layers8[metric_name]
    nu_layers16 = split_apply_combine_mean(nu_layers_memory[1])
    layers2 = nu_layers16[metric_name]
    # ----------- ACTFN
    act1 = split_apply_combine_mean(nu_actf_memory[0])
    tanh = act1[metric_name]
    act2 = split_apply_combine_mean(nu_actf_memory[1])
    relu = act2[metric_name]
    act3 = split_apply_combine_mean(nu_actf_memory[2])
    sigmoid = act3[metric_name]
    # ------------ OPT FUN
    opt1 = split_apply_combine_mean(nu_opt_memory[0])
    adam = opt1[metric_name]
    opt2 = split_apply_combine_mean(nu_opt_memory[1])
    rmsprop = opt2[metric_name]
    opt3 = split_apply_combine_mean(nu_opt_memory[2])
    adagrad = opt3[metric_name]
    opt4 = split_apply_combine_mean(nu_opt_memory[3])
    sdg = opt4[metric_name]
    # ------------ EPOCH
    E1 = split_apply_combine_mean(nu_epoch_memory[0])
    E10 = E1[metric_name]
    E2 = split_apply_combine_mean(nu_epoch_memory[1])
    E20 = E2[metric_name]
    E3 = split_apply_combine_mean(nu_epoch_memory[2])
    E30 = E3[metric_name]
    # ------------ LEARNING RATE
    rate01 = split_apply_combine_mean(nu_lrate_memory[0])
    rate01_ut = rate01[metric_name]
    rate03 = split_apply_combine_mean(nu_lrate_memory[1])
    rate03_ut = rate03[metric_name]
    rate06 = split_apply_combine_mean(nu_lrate_memory[2])
    rate06_ut = rate06[metric_name]
    # -------------- BATCH SIZE
    batch1 = split_apply_combine_mean(nu_batch_memory[0])
    b252 = batch1[metric_name]
    batch2 = split_apply_combine_mean(nu_batch_memory[1])
    b504 = batch2[metric_name]
    # -------------- NEG PAIRS
    neg1 = split_apply_combine_mean(nu_neg_memory[0])
    neg50 = neg1[metric_name]
    neg2 = split_apply_combine_mean(nu_neg_memory[1])
    neg50 = neg2[metric_name]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=4, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=1.1)
    plt.rcParams['ytick.labelsize'] = 'xx-small'
    plt.rcParams['xtick.labelsize'] = 'xx-small'
    # ------------ FACTORS
    p1 = ax[0, 0].errorbar(p, n8, yerr=error_calc(n8), label="8")
    p2 = ax[0, 0].errorbar(p, n16, yerr=error_calc(n16), label="16")
    p90 = ax[0, 0].errorbar(p, n24, yerr=error_calc(n24), label="24")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 0].set_title("Num Factors", fontsize='x-small')
    ax[0, 0].legend(handles=[p1, p2, p90], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LAYERS
    p3 = ax[0, 1].errorbar(p, layers1, yerr=error_calc(layers1), label="64:8")
    p4 = ax[0, 1].errorbar(p, layers2, yerr=error_calc(layers2), label="128:16")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 1].set_title("Layers", fontsize='x-small')
    ax[0, 1].legend(handles=[p3, p4], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ ACTFN
    p5 = ax[1, 0].errorbar(p, tanh, yerr=error_calc(tanh), label="tanh")
    p6 = ax[1, 0].errorbar(p, relu, yerr=error_calc(relu), label="relu")
    p7 = ax[1, 0].errorbar(p, sigmoid, yerr=error_calc(sigmoid), label="sig")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 0].set_title("Activation Function", fontsize='x-small')
    ax[1, 0].legend(handles=[p5, p6, p7], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ OPT FUN
    p8 = ax[1, 1].errorbar(p, adam, yerr=error_calc(adam), label="adam")
    p9 = ax[1, 1].errorbar(p, rmsprop, yerr=error_calc(rmsprop), label="rmsprop")
    p10 = ax[1, 1].errorbar(p, adagrad, yerr=error_calc(adagrad), label="adagrad")
    p11 = ax[1, 1].errorbar(p, sdg, yerr=error_calc(sdg), label="sdg")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 1].set_title("Optimisation Function", fontsize='x-small')
    ax[1, 1].legend(handles=[p8, p9, p10, p11], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- EPOCH
    p12 = ax[2, 0].errorbar(p, E10, yerr=error_calc(E10), label="10")
    p13 = ax[2, 0].errorbar(p, E20, yerr=error_calc(E20), label="20")
    p14 = ax[2, 0].errorbar(p, E30, yerr=error_calc(E30), label="30")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[2, 0].set_title("Epoch count", fontsize='x-small')
    ax[2, 0].legend(handles=[p12, p13, p14], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LEARN RATE
    p15 = ax[2, 1].errorbar(p, rate01_ut, yerr=error_calc(rate01_ut), label="0.001")
    p16 = ax[2, 1].errorbar(p, rate03_ut, yerr=error_calc(rate03_ut), label="0.002")
    p17 = ax[2, 1].errorbar(p, rate06_ut, yerr=error_calc(rate06_ut), label="0.006")
    ax[2, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[2, 1].set_title("learning rate", fontsize='x-small')
    ax[2, 1].legend(handles=[p15, p16, p17], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- BATCH SIZE
    p18 = ax[3, 0].errorbar(p, b252, yerr=error_calc(b252), label="256")
    p19 = ax[3, 0].errorbar(p, b504, yerr=error_calc(b504), label="512")
    ax[3, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[3, 0].set_title("Batch Size", fontsize='x-small')
    ax[3, 0].legend(handles=[p18, p19], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- NEG PAIRS
    p18 = ax[3, 1].errorbar(p, b252, yerr=error_calc(b252), label="50")
    p19 = ax[3, 1].errorbar(p, b504, yerr=error_calc(b504), label="100")
    ax[3, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[3, 1].set_title("Negative pairing", fontsize='x-small')
    ax[3, 1].legend(handles=[p18, p19], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    fig.suptitle(" Neural MF Model parameters and corresponding %s duration" % metric_name)
    fig.savefig('Results/Graphs/Neumf/' + dir_name + '/Memory/%s.png' % metric_name)
    plt.close(fig)


# method graphs NCF metrics data
# @param nu_factors_metrics: list containing num factors parameter model's data
#        nu_layers_metrics: list containing layers parameter model's data
#        nu_actf_metrics: list containing activation function parameter models's data
#        nu_opt_metrics: list containing optimisation function parameter model's data
#        nu_epoch_metrics: list containing epoch parameter model's data
#        nu_batch_metrics: list containing batch size parameter model's data
#        nu_lrate_metrics: list containing learn rate parameter model's data
#        nu_neg_metrics: list containing negative pairs parameter model's data
#        metric_name: string representing metric data column names
#        metric_title: string representing metric data column name for use in graph title
#        data_title: string denoting data size used in training/testing of the model
def nu_metrics(nu_factors_metrics, nu_layers_metrics, nu_actf_metrics, nu_opt_metrics,
               nu_epoch_metrics, nu_batch_metrics, nu_lrate_metrics, nu_neg_metrics, metric_name,
               metric_title, data_title):
    dir_name = data_title
    sub_plot_range = []
    if str(data_title) == '100k':
        auc_range = [0.6, 1]
        f1_range = [0, 0.03]
        mae_range = [2.4, 2.7]
        map_range = [0, 0.12]
        mrr_range = [0, 0.4]
        mse_range = [7, 8.4]
        ncrr_range = [0, 0.25]
        ndcg_range = [0.1, 0.6]
        prec_range = [0, 0.14]
        rec_range = [0.04, 0.13]
        rmse_range = [2.5, 2.9]
    else:
        auc_range = [0.6, 1]
        f1_range = [0, 0.03]
        mae_range = [2.4, 2.7]
        map_range = [0, 0.05]
        mrr_range = [0, 0.4]
        mse_range = [7, 8.4]
        ncrr_range = [0, 0.25]
        ndcg_range = [0.1, 0.6]
        prec_range = [0, 0.12]
        rec_range = [0.04, 0.11]
        rmse_range = [2.5, 2.9]
    if metric_name == 'auc':
        sub_plot_range = auc_range
    elif metric_name == 'f1-1':
        sub_plot_range = f1_range
    elif metric_name == 'mae':
        sub_plot_range = mae_range
    elif metric_name == 'map':
        sub_plot_range = map_range
    elif metric_name == 'mrr':
        sub_plot_range = mrr_range
    elif metric_name == 'mse':
        sub_plot_range = mse_range
    elif metric_name == 'ncrr':
        sub_plot_range = ncrr_range
    elif metric_name == 'ndcg':
        sub_plot_range = ndcg_range
    elif metric_name == 'precision':
        sub_plot_range = prec_range
    elif metric_name == 'recall':
        sub_plot_range = rec_range
    elif metric_name == 'rmse':
        sub_plot_range = rmse_range
    # --------------- TIME GRAPHS
    p = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # ----------- NUM FACT
    nu_fact_8 = split_apply_combine_mean(nu_factors_metrics[0])
    n8 = nu_fact_8[metric_name]
    nu_fact_16 = split_apply_combine_mean(nu_factors_metrics[1])
    n16 = nu_fact_16[metric_name]
    nu_fact_24 = split_apply_combine_mean(nu_factors_metrics[2])
    n24 = nu_fact_24[metric_name]
    # ----------- LAYERS
    nu_layers8 = split_apply_combine_mean(nu_layers_metrics[0])
    layers1 = nu_layers8[metric_name]
    nu_layers16 = split_apply_combine_mean(nu_layers_metrics[1])
    layers2 = nu_layers16[metric_name]
    # ----------- ACTFN
    act1 = split_apply_combine_mean(nu_actf_metrics[0])
    tanh = act1[metric_name]
    act2 = split_apply_combine_mean(nu_actf_metrics[1])
    relu = act2[metric_name]
    act3 = split_apply_combine_mean(nu_actf_metrics[2])
    sigmoid = act3[metric_name]
    # ------------ OPT FUN
    opt1 = split_apply_combine_mean(nu_opt_metrics[0])
    adam = opt1[metric_name]
    opt2 = split_apply_combine_mean(nu_opt_metrics[1])
    rmsprop = opt2[metric_name]
    opt3 = split_apply_combine_mean(nu_opt_metrics[2])
    adagrad = opt3[metric_name]
    opt4 = split_apply_combine_mean(nu_opt_metrics[3])
    sdg = opt4[metric_name]
    # ------------ EPOCH
    E1 = split_apply_combine_mean(nu_epoch_metrics[0])
    E10 = E1[metric_name]
    E2 = split_apply_combine_mean(nu_epoch_metrics[1])
    E20 = E2[metric_name]
    E3 = split_apply_combine_mean(nu_epoch_metrics[2])
    E30 = E3[metric_name]
    # ------------ LEARNING RATE
    rate01 = split_apply_combine_mean(nu_lrate_metrics[0])
    rate01_ut = rate01[metric_name]
    rate03 = split_apply_combine_mean(nu_lrate_metrics[1])
    rate03_ut = rate03[metric_name]
    rate06 = split_apply_combine_mean(nu_lrate_metrics[2])
    rate06_ut = rate06[metric_name]
    # -------------- BATCH SIZE
    batch1 = split_apply_combine_mean(nu_batch_metrics[0])
    b252 = batch1[metric_name]
    batch2 = split_apply_combine_mean(nu_batch_metrics[1])
    b504 = batch2[metric_name]
    # -------------- NEG PAIRS
    neg1 = split_apply_combine_mean(nu_neg_metrics[0])
    neg50 = neg1[metric_name]
    neg2 = split_apply_combine_mean(nu_neg_metrics[1])
    neg50 = neg2[metric_name]

    fontP = FontProperties()
    fontP.set_size('xx-small')
    fig, ax = plt.subplots(nrows=4, ncols=2)
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=1.1)
    plt.rcParams['ytick.labelsize'] = 'x-small'
    plt.rcParams['xtick.labelsize'] = 'x-small'
    # ------------ FACTORS
    p1 = ax[0, 0].errorbar(p, n8, yerr=error_calc(n8), label="8")
    p2 = ax[0, 0].errorbar(p, n16, yerr=error_calc(n16), label="16")
    p90 = ax[0, 0].errorbar(p, n24, yerr=error_calc(n24), label="24")
    ax[0, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 0].set_title("Num Factors", fontsize='x-small')
    ax[0, 0].legend(handles=[p1, p2, p90], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LAYERS
    p3 = ax[0, 1].errorbar(p, layers1, yerr=error_calc(layers1), label="64:8")
    p4 = ax[0, 1].errorbar(p, layers2, yerr=error_calc(layers2), label="128:16")
    ax[0, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[0, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[0, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[0, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[0, 1].set_title("Layers", fontsize='x-small')
    ax[0, 1].legend(handles=[p3, p4], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ ACTFN
    p5 = ax[1, 0].errorbar(p, tanh, yerr=error_calc(tanh), label="tanh")
    p6 = ax[1, 0].errorbar(p, relu, yerr=error_calc(relu), label="relu")
    p7 = ax[1, 0].errorbar(p, sigmoid, yerr=error_calc(sigmoid), label="sig")
    ax[1, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 0].set_title("Activation Function", fontsize='x-small')
    ax[1, 0].legend(handles=[p5, p6, p7], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ OPT FUN
    p8 = ax[1, 1].errorbar(p, adam, yerr=error_calc(adam), label="adam")
    p9 = ax[1, 1].errorbar(p, rmsprop, yerr=error_calc(rmsprop), label="rmsprop")
    p10 = ax[1, 1].errorbar(p, adagrad, yerr=error_calc(adagrad), label="adagrad")
    p11 = ax[1, 1].errorbar(p, sdg, yerr=error_calc(sdg), label="sdg")
    ax[1, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[1, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[1, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[1, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[1, 1].set_title("Optimisation Function", fontsize='x-small')
    ax[1, 1].legend(handles=[p8, p9, p10, p11], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # --------------- EPOCH
    p12 = ax[2, 0].errorbar(p, E10, yerr=error_calc(E10), label="10")
    p13 = ax[2, 0].errorbar(p, E20, yerr=error_calc(E20), label="20")
    p14 = ax[2, 0].errorbar(p, E30, yerr=error_calc(E30), label="30")
    ax[2, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[2, 0].set_title("Epoch count", fontsize='x-small')
    ax[2, 0].legend(handles=[p12, p13, p14], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # ------------ LEARN RATE
    p15 = ax[2, 1].errorbar(p, rate01_ut, yerr=error_calc(rate01_ut), label="0.001")
    p16 = ax[2, 1].errorbar(p, rate03_ut, yerr=error_calc(rate03_ut), label="0.002")
    p17 = ax[2, 1].errorbar(p, rate06_ut, yerr=error_calc(rate06_ut), label="0.006")
    ax[2, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[2, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[2, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[2, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[2, 1].set_title("learning rate", fontsize='x-small')
    ax[2, 1].legend(handles=[p15, p16, p17], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- BATCH SIZE
    p18 = ax[3, 0].errorbar(p, b252, yerr=error_calc(b252), label="256")
    p19 = ax[3, 0].errorbar(p, b504, yerr=error_calc(b504), label="512")
    ax[3, 0].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 0].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 0].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 0].set_ylabel(metric_name, fontsize='xx-small')
    ax[3, 0].set_title("Batch Size", fontsize='x-small')
    ax[3, 0].legend(handles=[p18, p19], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    # -------------- NEG PAIRS
    p18 = ax[3, 1].errorbar(p, b252, yerr=error_calc(b252), label="50")
    p19 = ax[3, 1].errorbar(p, b504, yerr=error_calc(b504), label="100")
    ax[3, 1].set_ylim([sub_plot_range[0], sub_plot_range[1]])
    ax[3, 1].set_xticks(np.arange(min(p), max(p) + 1, 10))
    ax[3, 1].set_xlabel("sample size (%)", fontsize='xx-small')
    ax[3, 1].set_ylabel(metric_name, fontsize='xx-small')
    ax[3, 1].set_title("Negative pairing", fontsize='x-small')
    ax[3, 1].legend(handles=[p18, p19], bbox_to_anchor=(1, 1), loc='upper left',
                    prop=fontP)
    fig.suptitle('Neural MF Model parameters and corresponding %s metrics' % metric_title)
    fig.savefig('Results/Graphs/Neumf/' + dir_name + '/Metrics/%s.png' % metric_name)
    plt.close(fig)


# method coordinates the function of the whole Analysis script
# reads all csv files produced by the rounds of experiments into seperate csv files,
# calls data_splitter() on each to split data into compartmentalised dataframes,
# dataframes then converted to series objects for graphing.
def runner():
    # data frame objects of each csv dataset
    before_time = pd.read_csv("before_time.csv")
    after_time = pd.read_csv("after_time.csv")
    time = pd.read_csv("Time_Recordings.csv")
    before_mem = pd.read_csv("before_system_memory_copy.csv")
    after_mem = pd.read_csv("after_system_memory_copy.csv")
    memory = pd.read_csv("Memory_Recordings.csv")
    metrics = pd.read_csv("Metric_Recordings.csv")

    # split data sets per model_parameter MF
    # ---------- MAX ITER data
    mf_iter_memory = data_splitter(memory)[0:3]
    mf_iter_metrics = data_splitter(metrics)[0:3]
    mf_iter_times = data_splitter(time)[0:3]
    bpr_iter_memory = data_splitter(memory)[18:21]  # ------------ BPR MAX ITER
    bpr_iter_metrics = data_splitter(metrics)[18:21]
    bpr_iter_time = data_splitter(time)[18:21]
    # 1 MILLION
    mf_mx_mem_1m = data_splitter(memory)[54:57]
    mf_mx_met_1m = data_splitter(metrics)[54:57]
    mf_mx_time_1m = data_splitter(time)[54:57]
    bpr_mx_mem_1m = data_splitter(memory)[72:75]
    bpr_mx_met_1m = data_splitter(metrics)[72:75]
    bpr_mx_time_1m = data_splitter(time)[72:75]
    # ---------- K MODELS data
    mf_k_memory = data_splitter(memory)[3:6]
    mf_k_metrics = data_splitter(metrics)[3:6]
    mf_k_times = data_splitter(time)[3:6]
    bpr_k_memory = data_splitter(memory)[21:24]  # -------------- BPR K MODELS
    bpr_k_metrics = data_splitter(metrics)[21:24]
    bpr_k_times = data_splitter(time)[21:24]
    # 1 MILLION
    mf_k_mem_1m = data_splitter(memory)[57:60]
    mf_k_met_1m = data_splitter(metrics)[57:60]
    mf_k_time_1m = data_splitter(time)[57:60]
    bpr_k_mem_1m = data_splitter(memory)[75:78]
    bpr_k_met_1m = data_splitter(metrics)[75:78]
    bpr_k_time_1m = data_splitter(time)[75:78]
    # ---------- LEARN RATE data
    mf_l_memory = data_splitter(memory)[6:9]
    mf_l_metrics = data_splitter(metrics)[6:9]
    mf_l_times = data_splitter(time)[6:9]
    bpr_l_memory = data_splitter(memory)[24:27]  # -------------- BPR LEARN RATE
    bpr_l_metrics = data_splitter(metrics)[24:27]
    bpr_l_time = data_splitter(time)[24:27]
    # 1 MILLION
    mf_lr_mem_1m = data_splitter(memory)[60:63]
    mf_lr_met_1m = data_splitter(metrics)[60:63]
    mf_lr_time_1m = data_splitter(time)[60:63]
    bpr_lr_mem_1m = data_splitter(memory)[78:81]
    bpr_lr_met_1m = data_splitter(metrics)[78:81]
    bpr_lr_time_1m = data_splitter(time)[78:81]
    # ---------- LAMBDA REG data
    mf_lam_reg_memory = data_splitter(memory)[9:12]
    mf_lam_reg_metrics = data_splitter(metrics)[9:12]
    mf_lam_reg_times = data_splitter(time)[9:12]
    bpr_lam_reg_memory = data_splitter(memory)[27:30]  # ---------- BPR LAM REG
    bpr_lam_reg_metrics = data_splitter(metrics)[27:30]
    bpr_lam_reg_time = data_splitter(time)[27:30]
    # 1 MILLION
    mf_lam_mem_1m = data_splitter(memory)[63:66]
    mf_lam_met_1m = data_splitter(metrics)[63:66]
    mf_lam_time_1m = data_splitter(time)[63:66]
    bpr_lam_mem_1m = data_splitter(memory)[81:84]
    bpr_lam_met_1m = data_splitter(metrics)[81:84]
    bpr_lam_time_1m = data_splitter(time)[81:84]
    # ---------- USE BIAS data
    mf_ub_memory = data_splitter(memory)[12:14]
    mf_ub_metrics = data_splitter(metrics)[12:14]
    mf_ub_time = data_splitter(time)[12:14]
    # 1 MILLION
    mf_ub_mem_1m = data_splitter(memory)[66:68]
    mf_ub_met_1m = data_splitter(metrics)[66:68]
    mf_ub_time_1m = data_splitter(time)[66:68]
    # ---------- EARLY STOP data
    mf_early_stop_memory = data_splitter(memory)[14:16]
    mf_early_stop_metrics = data_splitter(metrics)[14:16]
    mf_early_stop_time = data_splitter(time)[14:16]
    # 1 MILLION
    mf_es_mem_1m = data_splitter(memory)[68:70]
    mf_es_met_1m = data_splitter(metrics)[68:70]
    mf_es_time_1m = data_splitter(time)[68:70]
    # ---------- VERBOSE data
    mf_verbose_memory = data_splitter(memory)[16:18]
    mf_verbose_metrics = data_splitter(metrics)[16:18]
    mf_verbose_time = data_splitter(time)[16:18]
    bpr_verbose_memory = data_splitter(memory)[30:32]  # ------------ BPR VERBOSE
    bpr_verbose_metrics = data_splitter(metrics)[30:32]
    bpr_verbose_time = data_splitter(time)[30:32]
    # 1 MILLION
    mf_vb_mem_1m = data_splitter(memory)[70:72]
    mf_vb_met_1m = data_splitter(metrics)[70:72]
    mf_vb_time_1m = data_splitter(time)[70:72]
    bpr_vb_mem_1m = data_splitter(memory)[84:86]
    bpr_vb_met_1m = data_splitter(metrics)[84:86]
    bpr_vb_time_1m = data_splitter(time)[84:86]
    # ----------- NEUMF data
    # FACTORS
    nu_factors_memory = data_splitter(memory)[32:35]
    nu_factors_metrics = data_splitter(metrics)[32:35]
    nu_factors_time = data_splitter(time)[32:35]
    # LAYERS
    nu_layers_memory = data_splitter(memory)[35:37]
    nu_layers_metrics = data_splitter(metrics)[35:37]
    nu_layers_time = data_splitter(time)[35:37]
    # ACT FUNC
    nu_actf_memory = data_splitter(memory)[37:40]
    nu_actf_metrics = data_splitter(metrics)[37:40]
    nu_actf_time = data_splitter(time)[37:40]
    # OPTIMISER
    nu_opt_memory = data_splitter(memory)[40:44]
    nu_opt_metrics = data_splitter(metrics)[40:44]
    nu_opt_time = data_splitter(time)[40:44]
    # EPOCH
    nu_epoch_memory = data_splitter(memory)[44:47]
    nu_epoch_metrics = data_splitter(metrics)[44:47]
    nu_epoch_time = data_splitter(time)[44:47]
    # BATCH SIZE
    nu_batch_memory = data_splitter(memory)[47:49]
    nu_batch_metrics = data_splitter(metrics)[47:49]
    nu_batch_time = data_splitter(time)[47:49]
    # LEARN RATE
    nu_lrate_memory = data_splitter(memory)[49:52]
    nu_lrate_metrics = data_splitter(metrics)[49:52]
    nu_lrate_time = data_splitter(time)[49:52]
    # NEG_PAIR
    nu_neg_memory = data_splitter(memory)[52:54]
    nu_neg_metrics = data_splitter(metrics)[52:54]
    nu_neg_time = data_splitter(time)[52:54]

    # Graph Headers
    time_list = ["user time", "system time"]
    metric_list = ['mae', 'mse', 'rmse', 'auc', 'f1-1', 'map', 'mrr', 'ncrr', 'ndcg', 'precision', 'recall']
    metric_title_list = ["Mean Absolute Error", "Mean Squared Error", "Root Mean Squared Error", "Area Under Curve",
                         "F Measure", "Mean Average Precision", "Mean Reciprocal Rank",
                         "Normalised Cumulative Reciprocal Rank", "Normalised Discount Cumulative Gain",
                         "Precision", "Recall"]
    memory_stats = ['vmpeak(kb)', 'vmsize(kb)', 'vmlck(kb)', 'vmpin', 'vmhwm(kb)', 'vmrss(kb)', 'vmdata(kb)',
                    'vmstk(kb)', 'vmexe(kb)', 'vmlib(kb)', 'vmpte(kb)', 'vmswap(kb)']
    # # MF GRAPHS
    mf_time_100k_data = [mf_ub_time, mf_early_stop_time, mf_verbose_time, mf_iter_times, mf_k_times, mf_l_times,
                    mf_lam_reg_times]
    mf_metric_100k_data = [mf_iter_metrics, mf_k_metrics, mf_l_metrics, mf_lam_reg_metrics, mf_ub_metrics,
                      mf_early_stop_metrics, mf_verbose_metrics]
    mf_memory_100k_data = [mf_iter_memory, mf_k_memory, mf_l_memory, mf_lam_reg_memory, mf_ub_memory,
                      mf_early_stop_memory, mf_verbose_memory]
    mf_graph_maker(mf_time_100k_data, time_list, mf_metric_100k_data, metric_list, metric_title_list, mf_memory_100k_data,
                   memory_stats, '100k')
    mf_time_1m_data = [mf_ub_time_1m, mf_es_time_1m, mf_vb_time_1m, mf_mx_time_1m, mf_k_time_1m, mf_lr_time_1m,
                       mf_lam_time_1m]
    mf_metric_1m_data = [mf_mx_met_1m, mf_k_met_1m, mf_lr_met_1m, mf_lam_met_1m, mf_ub_met_1m, mf_es_met_1m,
                         mf_vb_met_1m]
    mf_memory_1m_data = [mf_mx_mem_1m, mf_k_mem_1m, mf_lr_mem_1m, mf_lam_mem_1m, mf_ub_mem_1m, mf_es_mem_1m,
                         mf_vb_mem_1m]
    mf_graph_maker(mf_time_1m_data, time_list, mf_metric_1m_data, metric_list, metric_title_list,
                   mf_memory_1m_data,
                   memory_stats, '1M')
    # # BPR GRAPHS
    bpr_time_100k_data = [bpr_iter_time, bpr_k_times, bpr_l_time, bpr_lam_reg_time, bpr_verbose_time]
    bpr_memory_100k_data = [bpr_iter_memory, bpr_k_memory, bpr_l_memory, bpr_lam_reg_memory, bpr_verbose_memory]
    bpr_metric_100k_data = [bpr_iter_metrics, bpr_k_metrics, bpr_l_metrics, bpr_lam_reg_metrics, bpr_verbose_metrics]
    bpr_graph_maker(bpr_time_100k_data, time_list, bpr_memory_100k_data, memory_stats, bpr_metric_100k_data,
                    metric_list, metric_title_list, '100k')
    bpr_time_1m_data = [bpr_mx_time_1m, bpr_k_time_1m, bpr_lr_time_1m, bpr_lam_time_1m, bpr_vb_time_1m]
    bpr_memory_1m_data = [bpr_mx_mem_1m, bpr_k_mem_1m, bpr_lr_mem_1m, bpr_lam_mem_1m, bpr_vb_mem_1m]
    bpr_metric_1m_data = [bpr_mx_met_1m, bpr_k_met_1m, bpr_lr_met_1m, bpr_lam_met_1m, bpr_vb_met_1m]
    bpr_graph_maker(bpr_time_1m_data, time_list, bpr_memory_1m_data, memory_stats, bpr_metric_1m_data,
                    metric_list, metric_title_list, '1M')
    # # # NEUMF Graphs
    nu_time_100k_data = [nu_factors_time, nu_layers_time, nu_actf_time, nu_opt_time,
                    nu_epoch_time, nu_batch_time, nu_lrate_time, nu_neg_time]
    nu_memory_100k_data = [nu_factors_memory, nu_layers_memory, nu_actf_memory, nu_opt_memory,
                      nu_epoch_memory, nu_batch_memory, nu_lrate_memory, nu_neg_memory]
    nu_metric_100k_data = [nu_factors_metrics, nu_layers_metrics, nu_actf_metrics, nu_opt_metrics,
                      nu_epoch_metrics, nu_batch_metrics, nu_lrate_metrics, nu_neg_metrics]
    nu_graph_maker(nu_time_100k_data, time_list, nu_memory_100k_data, memory_stats, nu_metric_100k_data,
                   metric_list, metric_title_list, '100k')


runner()
