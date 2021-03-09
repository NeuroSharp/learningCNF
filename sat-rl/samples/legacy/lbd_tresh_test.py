import sys, os
import logging

from pysat.solvers   import Minisat22
from pysat.formula       import CNF
from pysat.callbacks import empty_cb

import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

import numpy as np
import time

import csv

from collections import Counter

log = logging.getLogger(__name__)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')

rewards = []
gc_count = 0
init_r = 0
lbd_thresh = 1
range_min = 1
range_max = 3#31


def save_plot(fname, x_lbd, lbd_reward):
    df_lbd = pd.DataFrame(columns=['thresh', 'gc_calls', 'reward'], data=x_lbd)
    df_lbd.set_index('thresh', inplace=True)

    ax = df_lbd[['reward']].plot.line(grid=True,
                                                                      xticks=range(range_min, range_max, 2),
                                                                      title=fname)

    ax.hlines(y=lbd_reward, xmin=range_min, xmax=range_max, linewidth=1, color='r', linestyle='-.', label="LBD")
    ax2 = df_lbd[['gc_calls']].plot.line(ax = ax, secondary_y=True, grid=True)
    ax.set_xlabel("LBD Threshold")
    ax.set_ylabel('Reward')
    ax2.set_ylabel('# of GC Calls')

    plt.savefig(fname + '.png')
    plt.close()

def save_summary_plot(fname, plot_name, df):
    # range_max = 23
    stats = df.loc[['mean', 'std']].iloc[:,2:range_max+1].T
    stats.index = np.arange(range_min, range_max)

    ax = stats['mean'].plot.line(grid=True, xticks=range(1, range_max, 1), title=plot_name, label=r'$mean(LBD_{fixed})$')
    ax.fill_between(range(range_min, range_max),
            stats['mean'] - stats['std'],
            stats['mean'] + stats['std'], color='b', alpha=0.2)

    trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData)

    ax.hlines(y=df.loc['mean']['lbd_reward'], xmin=range_min, xmax=range_max-1, linewidth=1, color='r', linestyle='-.', label=r'$LBD_{orig}$')
    ax.text(0,df.loc['mean']['lbd_reward'], "{:.2f}".format(df.loc['mean']['lbd_reward']), transform=trans, color='r', ha="right", va="center")
    ax.hlines(y=df.loc['mean']['max_thresh_reward'], xmin=range_min, xmax=range_max-1, linewidth=1, color='y', linestyle='-',  label=r'$max(LBD_{fixed})$')
    ax.text(0,df.loc['mean']['max_thresh_reward'], "{:.2f}".format(df.loc['mean']['max_thresh_reward']), transform=trans, color='y', ha="right", va="center")
    # plt.fill_between(range(range_min, range_max),
    #       df.loc['mean']['max_thresh_reward'] - df.loc['std']['max_thresh_reward'],
    #       df.loc['mean']['max_thresh_reward'] + df.loc['std']['max_thresh_reward'], color='y', linestyle='-.', alpha=0.2)

    ax.set_xlabel("LBD Threshold")
    ax.set_ylabel('Reward')
    ax.legend()

    plt.xticks(rotation=90)
    # plt.yticks(np.arange(0, 1, step=0.2))
    plt.savefig(fname + '.svg', format="svg")
    plt.close()

def save_hist(fname, plot_name, df):
    df = df.iloc[:, 3:]
    df.columns = range(range_min, range_max)
    df = df.idxmax(axis = 1)

    counts = dict(sorted(Counter(df).items()))
    ax = pd.DataFrame.from_dict(counts, orient='index').plot(kind='bar', legend=False, title=plot_name)

    ax.set_xlabel("LBD Threshold")
    ax.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(fname + '.hist.svg', format="svg")
    plt.close()

def thresh_cb(cl_label_arr, rows_arr, cols_arr, data_arr):
    global gc_count
    print('.', end='', flush=True)
    gc_count += 1

    lbd_cond = np.greater_equal(cl_label_arr[:,3], lbd_thresh)
    locked_cond = np.logical_not(cl_label_arr[:,5])

    return np.logical_not(np.logical_and(lbd_cond, locked_cond))

def empty_callback(cl_label_arr, rows_arr, cols_arr, data_arr):
    global gc_count
    global init_r
    print('.', end='', flush=True)
    gc_count += 1

    if(gc_count == 1):
        init_r = m.reward()

    return empty_cb(cl_label_arr, rows_arr, cols_arr, data_arr)

if __name__ == '__main__':
# def nothing():
    if(len(sys.argv) < 1 or not os.path.isdir(sys.argv[1])):
        print("usage: {} input_dir".format(sys.argv[0]))
        print("input_dir: The directory containing synthesized SAT instances.")
        exit(1)

    dir_path = sys.argv[1]
    folder_n = os.path.basename(os.path.normpath(dir_path))
    output_name = os.path.join(dir_path, folder_n)

    log.setLevel(logging.INFO)
    file_handler = logging.FileHandler("{}.log".format(output_name))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)


    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and
                                                                                            (f.endswith(".bz2") or f.endswith(".cnf"))]

    rewards = []
    for fname in files:
        m = Minisat22(callback=empty_callback, reduce_base=2000)
        fname = os.path.join(dir_path, fname)

        try:
            log.info("Reading: {}".format(fname))
            f1 = CNF(from_file='' + fname)
            m.append_formula(f1.clauses)

            gc_count = 0
            log.info("Solving: {}".format(fname))
            print(m.solve())
        except Exception as e:
            log.exception("Aborting {} due to exception: {}".format(fname, e))
            continue
        else:
            if (gc_count <= 1): # Low GC calls, drop the instance
                log.info("Aborting {} due to low # of GC calls.".format(fname))
                continue

            lbd_reward = 10 - (m.reward() - init_r) / 10 ** 7
            log.info("LBD reward: {}".format(lbd_reward))
        finally:
            m.delete()


        stats = np.empty((0,3), int)
        for lbdt in range(range_min, range_max):
            lbd_thresh = lbdt
            th_reward = 0

            m = Minisat22(callback=thresh_cb, reduce_base=2000)
            try:
                m.append_formula(f1.clauses)

                gc_count = 0
                init_r = 0 # Might not be necessary as it will get set in the callback
                log.info("Solving with lbd threshold: {}".format(lbd_thresh))
                print(m.solve())
            except Exception as e:
                log.exception("Aborting {} with threshold {} due to exception: {}".format(fname, e))
                break #out of the lbd threshold loop and continue to the next file
            else:
                stats = np.append(stats,
                        [[lbdt, gc_count, 10 - (m.reward() - init_r) / 10 ** 7]],
                        axis=0)
            finally:
                m.delete()
        else: # all lbd threshold runs were successful. Collect the results
            this_rewards = [fname, lbd_reward, max(stats[:, 2])] + stats[:, 2].tolist()
            log.info(this_rewards)
            rewards.append(this_rewards)

            save_plot(fname, stats, lbd_reward)

    log.info(rewards)

    df = pd.DataFrame(rewards, columns=['file_name', 'lbd_reward', 'max_thresh_reward'] + ["th_{}_reward".format(th) for th in range(range_min, range_max)])
    save_hist(output_name, "Best Fixed Threshold Histogram", df)

    df.set_index('file_name', inplace=True)
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    save_summary_plot(output_name, "Aggregated Rewards", df)

    df.to_csv("{}.csv".format(output_name))


# df = pd.read_csv('test/90batch/GoodPile90.csv')
# df = df.iloc[:-2, :]
# save_hist("GoodPile90", "Best Fixed Threshold Histogram", df)

# df = pd.read_csv('test/90batch/GoodPile90.csv')
# df.set_index('file_name', inplace=True)
# save_summary_plot("GoodPile90", "Aggregated Rewards", df)
