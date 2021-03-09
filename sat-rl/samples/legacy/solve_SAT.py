import os, sys, psutil

import multiprocessing as mp

import itertools
from contextlib import suppress
import pickle
import logging
import time

from pysat.solvers import Lingeling, Minisat22
from pysat.formula import CNF

from console_utils import print_progress

log = logging.getLogger(__name__)
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')

def solve(file, mq, pq):
    log.info("Solving: {}, pid: {}".format(file, os.getpid()))
    m = Lingeling (use_timer=True, time_lim=18000)

    try:
        f1 = CNF()
        f1.from_file(os.path.join(dir_path, file))

        m.append_formula(f1.clauses)
        res = m.solve()
    except Exception:
        log.exception("Aborting {} due to exception, pid: {}".format(file, os.getpid()))
        pq.put("");
    else:
        if(res == 0):
            log.info("Aborting {} due to timeout, pid: {}".format(file, os.getpid()))
            pq.put("");
        elif(res == 10):
            log.info("[SAT] Writing model {}..., pid: {}".format(file, os.getpid()))
            mq.put("{}, {}".format(file, m.time()))
            pq.put("");

            with open(os.path.join(res_path, "{}.pickle".format(file)), 'wb') as handle:
                pickle.dump(m.get_model(), handle, protocol=pickle.HIGHEST_PROTOCOL)
        else: # unsat
            log.info("[UNSAT] Ignoring {}..., pid: {}".format(file, os.getpid()))
            pq.put("");
    finally:
        m.delete()

        process = psutil.Process(os.getpid())
        log.info("Finishing the task {0}..., pid: {1}, memory: {2:.2f}MB".format(file, os.getpid(), process.memory_info().rss / float(2 ** 20)))

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
        print("usage: {} [input_dir]".format(sys.argv[0]))
        print("input_dir: The directory containing SAT instances.")
        exit(1)

    dir_path = sys.argv[1]
    res_path = os.path.join(dir_path, "models")

    log.setLevel(logging.INFO)
    file_handler = logging.FileHandler("{}.log".format(sys.argv[1]))
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)

    with suppress(OSError):
        os.mkdir(res_path)
    res_file = os.path.join(res_path, 'result.csv')

    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and
                                                                                    (f.endswith(".bz2") or f.endswith(".cnf"))]

    manager = mp.Manager()
    mq = manager.Queue() # message quque
    pq = manager.Queue() # progress queue

    pool = mp.Pool(processes=mp.cpu_count()-5, maxtasksperchild=1000)
    pool.apply_async(listener, (res_file, mq))
    pool.apply_async(progress, (pq, len(files)))

    jobs = []
    for file in files:
        job = pool.apply_async(solve, (file, mq, pq))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs:
        job.get()

    mq.put('kill')
    pq.put('kill')

    pool.close()
    pool.join()

    log.info("done!")
