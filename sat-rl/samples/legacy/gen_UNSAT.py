import os, sys, psutil

import multiprocessing as mp

import itertools
from contextlib import suppress
import logging
import time
import random
import numpy as np

from pysat.solvers import Glucose3
from pysat.formula import CNF

from console_utils import print_progress

import argparse

log = logging.getLogger(__name__)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')


def solve(file, index, end, mq, pq):
    key = random.randint(10000, 99999)
    np.random.seed(key) # reseeding the numpy random num gen
    f_name = file[:-4] # The resulting synth cnf file name
    if (f_name.endswith(".cnf")): f_name = f_name[:-4]
    f_name = "{}[{}-{}]-{}.cnf.bz2".format(f_name, index, end, key)

    log.info("Reading: {}, pid: {}".format(f_name, os.getpid()))
    cnf = CNF()

    m = Glucose3(gc_freq="fixed", use_timer=True)
    try:
        cnf.from_file(os.path.join(dir_path, file))
        m.append_formula(cnf.clauses)

        # find the list of variables that are not forced by the initial unit propagation (m.nof_vars())
        unforced_vars = [i for (i, f) in enumerate(m.get_var_labels()[0:, 5]) if f == 0]
        var_ass = np.random.choice(unforced_vars, index, replace=False) * np.random.choice([-1, 1], index)
        # append the units to the solver
        units = CNF()
        units.from_clauses([[int(x)] for x in var_ass])
        m.append_formula(units.clauses)

        m.time_budget(50)
        log.info("Solving: {} [{}-{}]-{}, unit#: {}, pid: {}".format(f_name, index, end, key, index, os.getpid()))
        res = m.solve_limited() # Capture res to check for None's for timeouts
    except Exception:
        log.exception("Aborting {} due to exception, pid: {}".format(f_name, os.getpid()))
        pq.put("");
    else:
        if(res == None): # No timeouts should happen
            log.error("Aborting {} due to timeout. # of GC calls {}, pid: {}".format(f_name, m.nof_gc(), os.getpid()))
            pq.put("");
        elif(not res): # UNSAT formula
            if (2 <= m.nof_gc() and m.nof_gc() <= 30):#3 <= m.time() and m.time() <= 10 and
                log.info("[UNSAT] Writing file {}, pid: {}".format(f_name, os.getpid()))

                f_addr = os.path.join(res_path, f_name)
                comments = ["c Name: {}".format(f_name),
                                        "c Time to solve (expected): {:.2f}s".format(m.time())]
                cnf.extend(units.clauses)
                cnf.to_file(f_addr, comments, compress_with="bzip2")

                mq.put("{}, {:.2f}s".format(f_name, m.time()))
                pq.put("");
            elif (m.nof_gc() < 2):
                log.info("Aborting {}! # of GC calls < 2, pid: {}".format(f_name, os.getpid()))
                pq.put("");
            else:
                log.info("Aborting {}! # of GC calls > 30, pid: {}".format(f_name, os.getpid()))
                # log.info("Aborting {}! Solving time outside of the time limits: {:.2f}s, pid: {}".format(f_name, m.time(), os.getpid()))
                pq.put("");
        else: # sat
            log.error("[SAT] Ignoring {}..., pid: {}".format(f_name, os.getpid()))
            pq.put("");
    finally:
        m.delete()
        del cnf

        process = psutil.Process(os.getpid())
        log.info("Finishing the task {0}..., pid: {1}, memory: {2:.2f}MB".format(f_name, os.getpid(), process.memory_info().rss / float(2 ** 20)))

def progress(pq, num_files):
    log.info("Progress bar started at pid:{}".format(os.getpid()))
    instance_cnt = 0

    while 1:
        m = pq.get()
        if m == 'kill':
            break

        instance_cnt += 1
        print_progress(instance_cnt, num_files, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)

    log.info("Progress bar stopped at pid:{}".format(os.getpid()))

def listener(fn, mq):
    '''listens for messages on the q, writes to file. '''
    log.info("Listener started at pid:{}".format(os.getpid()))

    f = open(fn, 'ab')

    while 1:
        m = mq.get()
        if m == 'kill':
            break

        message = str(m) + '\n'
        f.write(message.encode('utf-8'))
        f.flush()
    f.close()

    log.info("Listener stopped at pid:{}".format(os.getpid()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates UNSAT instances from a given benchmark.')
    parser.add_argument('dir_path', metavar='benchmark_path', type=str)
    # parser.add_argument('--n_splits', metavar='n_splits', default='1000')
    parser.add_argument('--begin', metavar='begin', default='1')
    parser.add_argument('--end', metavar='end', default='10')
    parser.add_argument('--increment', metavar='increment', default='1')
    parser.add_argument('--n_shuffles', metavar='n_shuff', default='100')
    parser.add_argument('--cores', metavar='cores', default=str(psutil.cpu_count(logical=False) - 1))
    args = parser.parse_args()

    dir_path = args.dir_path
    res_path = os.path.join(dir_path, "synth")

    # n_splits = int(args.n_splits)
    n_shuffles = int(args.n_shuffles)
    begin = int(args.begin)
    end = int(args.end)
    increment = int(args.increment)
    MAX_PROC = int(args.cores) - 3

    log.setLevel(logging.INFO)

    folder_name = os.path.basename(os.path.normpath(dir_path))
    file_handler = logging.FileHandler(os.path.join(dir_path, "{}-synth.log".format(folder_name)))
    file_handler.setFormatter(log_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)

    with suppress(OSError):
        os.mkdir(res_path)
    key = random.randint(10000, 99999)
    res_file = os.path.join(res_path, 'result-gen{}.csv'.format(key))

    synth_files = list(set([f.split("[")[0] for f in os.listdir(res_path)]))
    cnf_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and
                                                                                    (f.endswith(".bz2") or f.endswith(".cnf")) and
                                                                                    (not f.split(".cnf")[0] in synth_files)]

    random.shuffle(cnf_files)
    log.info("Working with {} original formulas.".format(len(cnf_files)));

    echelon = end - begin # int(cnf.nv*0.01)

    manager = mp.Manager()
    maxtasksperchild = 100
    mq = manager.Queue() # message quque
    pq = manager.Queue() # progress queue

    pool = mp.Pool(processes=MAX_PROC, maxtasksperchild=maxtasksperchild)
    pool.apply_async(listener, (res_file, mq))
    pool.apply_async(progress, (pq, len(cnf_files)*echelon*n_shuffles))

    jobs = []
    for cnf_f in cnf_files:
        for (i, shuff) in itertools.product(range(begin, end, increment), range(n_shuffles)):
            # solve(file, index, n_splits, mq, pq):
            job = pool.apply_async(solve, (cnf_f, i, end, mq, pq))
            jobs.append(job)

            while (len(jobs) == MAX_PROC):
                for job_id, job in reversed(list(enumerate(jobs))):
                    if (job.ready()):
                        job.get()
                        del jobs[job_id]

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    mq.put('kill')
    pq.put('kill')

    pool.close()
    pool.join()

    log.info("done!")
