import random
import logging
import time
import os, sys, psutil
import multiprocessing as mp
import argparse
from contextlib import suppress
import itertools
import traceback

from ecarev_gen import Benchmark
from sharpsat_filter import SharpSATFilter

from pysat.solvers import SharpSAT
from pysat.formula import CNF

log = logging.getLogger("sharpSAT")
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')

def listener(fn, mq):
    '''listens for messages on the q, writes to file. '''
    log.info("Listener started at pid:{}".format(os.getpid()))

    f = open(fn, 'ab')
    header = "file_name,Rule,n,f,r,s,var_len,cla_len,reward,time,model_count\n"
    f.write(header.encode('utf-8'))
    f.flush()

    while 1:
        m = mq.get()
        if m == 'kill':
            break

        message = str(m) + '\n'
        f.write(message.encode('utf-8'))
        f.flush()
    f.close()

    log.info("Listener stopped at pid:{}".format(os.getpid()))

def solve(R, n, f, r, s, cnt, mq=None): # mq: Message Queue is just for the parallel case
    process = psutil.Process(os.getpid())

    cnf_id = "ecarev-%i-%i-%i-%i-%i" % (R, n, f, r, cnt)
    file_path = os.path.join(res_path,"{}.cnf".format(cnf_id))
    Benchmark.print_dimacs(Benchmark(R, n, f, r, s),
        open(file_path, "w"))

    log.info(f"{cnf_id}: Starting! (pid: {os.getpid()}, memory: {process.memory_info().rss / float(2 ** 20):.2f}MB)")

    sharpSAT = SharpSAT(time_budget = time_max, use_timer= True)
    count = sharpSAT.solve(file_path)

    res = None
    if (not filter.filter(sharpSAT, cnf_id)):
        log.info(f"{cnf_id}: Saving! (pid: {os.getpid()}, memory: {process.memory_info().rss / float(2 ** 20):.2f}MB)")

        message = (f"{cnf_id}.cnf, {R}, {n}, {f}, {r}, {s}, {sharpSAT.nof_vars()}, "
        f"{sharpSAT.nof_clauses()}, {sharpSAT.reward()}, {sharpSAT.time():.2f}, {count}")
        if (mq): mq.put(message)

        res = count
    else:
        os.remove(file_path)

    sharpSAT.delete()

    log.info(f"{cnf_id}: Finishing! (pid: {os.getpid()}, memory: {process.memory_info().rss / float(2 ** 20):.2f}MB)")

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Generates a Cellular Automata benchmark_path."""
        , epilog="Example")
    parser.add_argument('--R_min', metavar='R_min', type=int, default=38)#26)
    parser.add_argument('--R_max', metavar='R_max', type=int, default=39) # pow(2, 8)
    parser.add_argument('--n_min', metavar='n_min', type=int, default=300)
    parser.add_argument('--n_max', metavar='n_max', type=int, default=400)
    parser.add_argument('--n_inc', metavar='n_inc', type=int, default=1)
    parser.add_argument('--f_min', metavar='f_min', type=int, default=11)
    parser.add_argument('--f_max', metavar='f_max', type=int, default=30)
    parser.add_argument('--f_inc', metavar='f_inc', type=int, default=1)
    parser.add_argument('--r_min', metavar='r_min', type=int, default=11)
    parser.add_argument('--r_max', metavar='r_max', type=int, default=30)
    parser.add_argument('--r_inc', metavar='r_inc', type=int, default=1)
    parser.add_argument('--max_samples', metavar='max_samples', type=int, default=2000)
    args = parser.parse_args()

    steps_min = 30
    time_max = 2
    time_min = 0.15
    filter = SharpSATFilter(steps_min, time_min, time_max)

    R_range = range(args.R_min, args.R_max + 1)
    n_range = range(args.n_min, args.n_max + 1, args.n_inc)
    f_range = range(args.f_min, args.f_max + 1, args.f_inc)
    r_range = range(args.r_min, args.r_max + 1, args.r_inc)

    sample_easy = 100

    MAX_PROC = psutil.cpu_count(logical=False) - 2 # No. of physical cores (leave 2)

    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset/sharpsat")
    with suppress(OSError):
        os.mkdir(res_path)
    res_file = os.path.join(res_path, 'range[{}-{}].csv'.format(args.R_min, args.R_max))


    log.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(res_path, "range[{}-{}].log".format(args.R_min, args.R_max)))
    file_handler.setFormatter(log_formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)

    log.addHandler(file_handler)
    log.addHandler(stream_handler)


    manager = mp.Manager()
    maxtasksperchild = 40
    mq = manager.Queue() # message quque

    pool = mp.Pool(processes=MAX_PROC, maxtasksperchild=maxtasksperchild)
    pool.apply_async(listener, (res_file, mq))

    jobs = []
    sample_cnt = 0
    try:
        for R in R_range:
            for n in n_range:
                for f in f_range:
                    for r in r_range:
                        if (sample_cnt > args.max_samples):
                            raise Exception("Done sampling")

                        log.info("Checking: R(%i)-n(%i)-f(%i)-r(%i)" % (R, n, f, r))
                        cnf_id = "ecarev-%i-%i-%i-%i" % (R, n, f, r)
                        file_path = os.path.join(res_path,"{}.cnf".format(cnf_id))
                        Benchmark.print_dimacs(Benchmark(R, n, f, r, random.randint(1,10)),
                            open(file_path, "w"))

                        sharpSAT = SharpSAT(time_budget = time_max, use_timer= True)
                        res = sharpSAT.solve(file_path)

                        is_degen = filter.filter(sharpSAT, cnf_id)
                        sharpSAT.delete()
                        os.remove(file_path)
                        if (is_degen):
                            continue
                        if (res == None):
                            log.info("Timed out!")
                            continue

                        log.info("Candidate Found: R(%i)-n(%i)-f(%i)-r(%i)" % (R, n, f, r))

                        for cnt in range(20, sample_easy + 21):
                            s = random.randrange(2, pow(2, n))
                            job = pool.apply_async(solve, (R, n, f, r, s, cnt, mq))
                            jobs.append(job)

                            while (len(jobs) == MAX_PROC):
                                for cnf_id, job in reversed(list(enumerate(jobs))):
                                    if (job.ready()):
                                        result = job.get()
                                        if (result != None): sample_cnt += 1
                                        del jobs[cnf_id]
    except:
        log.info(f"Reached max_samples of {args.max_samples}: {sample_cnt}")
    finally:
        # collect results from the workers through the pool result queue
        for job in jobs:
            job.get()

        mq.put('kill')

        pool.close()
        pool.join()

        log.info("done!")
