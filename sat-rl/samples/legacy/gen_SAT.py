import os, sys, psutil

import multiprocessing as mp

import itertools
from contextlib import suppress
import pickle
import logging
import time
import random

from pysat.solvers import Lingeling, Minisat22
from pysat.formula import CNF

from console_utils import print_progress

log = logging.getLogger(__name__)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')

def solve(file, model, index, n_splits, mq, pq):
    key = random.randint(10000, 99999)
    chunk = index*int(len(model)/n_splits)
    random.shuffle(model)
    var_ass = model[0:chunk]

    log.info("Solving: {} [{}-{}]-{}, unit#: {}, pid: {}".format(file, index, n_splits, key, len(var_ass), os.getpid()))
    m = Lingeling (use_timer=True, time_lim=20)

    try:
        f_name = file[:-4] # The resulting synth cnf file name
        if (f_name.endswith(".cnf")): f_name = f_name[:-4]
        f_name = "{}[{}-{}]-{}.cnf.bz2".format(f_name, index, n_splits, key)

        # read the actual cnf
        cnf = CNF()
        cnf.from_file(os.path.join(dir_path, file))

        # append the units to the read clauses from f1
        units = CNF()
        units.from_clauses([[x] for x in var_ass])
        cnf.extend(units.clauses)

        m.append_formula(cnf.clauses)
        res = m.solve()
    except Exception:
        log.exception("Aborting {} due to exception, pid: {}".format(f_name, os.getpid()))
        pq.put("");
    else:
        if(res == 0): # No timeouts should happen
            log.error("Aborting {} due to timeout, pid: {}".format(f_name, os.getpid()))
            pq.put("");
        elif(res == 10):
            if (3 <= m.time() and m.time() <= 10):
                log.info("[SAT] Writing file {}, pid: {}".format(f_name, os.getpid()))

                f_addr = os.path.join(res_path, f_name)
                comments = ["c Name: {}".format(f_name),
                                        "c Time to solve (expected): {:.2f}s".format(m.time())]
                cnf.to_file(f_addr, comments, compress_with="bzip2")
                mq.put("{}, {:.2f}s".format(f_name, m.time()))
                pq.put("");

            else:
                log.info("Aborting {}! Solving time outside of the time limits: {:.2f}s, pid: {}".format(f_name, m.time(), os.getpid()))
                pq.put("");
        else: # unsat
            log.error("[UNSAT] Ignoring {}..., pid: {}".format(f_name, os.getpid()))
            pq.put("");
    finally:
        m.delete()

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
    if(len(sys.argv) < 2):
        print("usage: {} input_dir [n_splits]".format(sys.argv[0]))
        print("input_dir: The directory containing SAT instances.")
        print("n_splits: The number of chunk the dataset is divided to. (default=1000)")
        exit(1)

    dir_path = sys.argv[1]
    models_path = os.path.join(dir_path, "models")
    res_path = os.path.join(models_path, "synth")

    n_splits = 1000 # Divide the model to this many chunks
    with suppress(Exception):
        n_splits = int(sys.argv[2])

    log.setLevel(logging.INFO)
    folder_name = os.path.basename(os.path.normpath(dir_path))
    file_handler = logging.FileHandler(os.path.join(dir_path, "{}-synth.log".format(folder_name)))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)

    with suppress(OSError):
        os.mkdir(res_path)
    res_file = os.path.join(res_path, 'result-gen.csv')

    pickle_files = [f for f in os.listdir(models_path) if os.path.isfile(os.path.join(models_path, f)) and
                                                                                    (f.endswith(".pickle"))]

    cnf_files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and
                                                                                    (f.endswith(".bz2") or f.endswith(".cnf"))]

    manager = mp.Manager()
    maxtasksperchild = 100
    mq = manager.Queue() # message quque
    pq = manager.Queue() # progress queue

    pool = mp.Pool(processes=mp.cpu_count()-5, maxtasksperchild=maxtasksperchild)
    pool.apply_async(listener, (res_file, mq))
    pool.apply_async(progress, (pq, len(pickle_files)*n_splits))

    jobs = []
    for pf in pickle_files:
        # NOTE: Use Minisat NOT Lingeling
        # 1. Find the associated cnf file
        cnfs = [f for f in cnf_files if f.startswith(pf[:-len(".pickle")])]
        if len(cnfs) < 1: # assiciated cnf file is not found
            log.error("The associated cnf file is not found for: {}".format(pf))
            continue
        cf = cnfs[0]

        # 2. unpickle the model
        with open(os.path.join(models_path, pf), 'rb') as handle:
            model = pickle.load(handle)

        # 3. Ask each worker to work on a random subset of the model
            if int(len(model)/n_splits) < 1: # Avoid having chunks of size 0
                n_splits = len(model)

            for i in range(0, n_splits-1):
                job = pool.apply_async(solve, (cf, model, i, n_splits, mq, pq))
                jobs.append(job)

                if(len(jobs) == maxtasksperchild):
                    # collect results from the workers through the pool result queue
                    for job in jobs:
                        job.get()

                    jobs = []

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    mq.put('kill')
    pq.put('kill')

    pool.close()
    pool.join()

    log.info("done!")
