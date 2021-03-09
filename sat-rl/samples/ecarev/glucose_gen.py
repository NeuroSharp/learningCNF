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
from glucose_filter import GlucoseFilter

from pysat.solvers import Glucose3
from pysat.formula import CNF

log = logging.getLogger("glucose")
log_formatter = logging.Formatter('%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s',
                                  datefmt='%d/%m/%Y %H:%M:%S')

def listener(fn, mq):
    '''listens for messages on the q, writes to file. '''
    log.info("Listener started at pid:{}".format(os.getpid()))

    f = open(fn, 'ab')
    header = "file_name,Rule,n,f,r,s,var_len,cla_len,gc_cnt,op_cnt,time,result\n"
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

def solve(R, n, f, r, s, cnt, reduce_base, mq=None, save=True): # mq: Message Queue is just for the parallel case
    process = psutil.Process(os.getpid())
    instance = Benchmark(R, n, f, r, s)
    cnf_id = "ecarev-%i-%i-%i-%i-%i" % (R, n, f, r, cnt)
    log.info("{0}: Starting! (pid: {1}, memory: {2:.2f}MB)".format(cnf_id, os.getpid(), process.memory_info().rss / float(2 ** 20)))

    glucose = Glucose3(gc_oracle = {"callback": lambda *args: None, "policy":"glucose"}, gc_freq="fixed", reduce_base=reduce_base)
    glucose.append_formula(instance.clauses)
    glucose.time_budget(time_max)
    res  = glucose.solve_limited()
    time = glucose.time() if (res != None) else time_max

    result = ""
    if (res == True): result = "SAT"
    elif (res == False): result = "UNSAT"
    else: result = "UNKNOWN"

    res = None
    if(not filter.filter(glucose, cnf_id)):
        log.info("{0}: Saving! (pid: {1}, memory: {2:.2f}MB)".format(cnf_id, os.getpid(), process.memory_info().rss / float(2 ** 20)))
        #         "file_name, R, n, r, f, s, var_len, cla_len, gc_cnt, op_cnt, time, result"
        message = "{}.cnf, {}, {}, {}, {}, {}, {}, {}, {}, {}, {:.2f}, {}".format(cnf_id, R, n, f, r, s,
            glucose.nof_vars(), glucose.nof_clauses(), glucose.nof_gc(), glucose.reward(), glucose.time(), result)
        if (mq): mq.put(message)

        cnf = CNF(from_clauses=instance.clauses)
        res = {"cnf": cnf, "steps": glucose.nof_gc()}
        if (save): cnf.to_file(os.path.join(res_path,"{}.cnf.bz2".format(cnf_id)), ["c " + message], compress_with="bzip2")

    glucose.delete()
    log.info("{0}: Finishing! (pid: {1}, memory: {2:.2f}MB)".format(cnf_id, os.getpid(), process.memory_info().rss / float(2 ** 20)))

    return res


def ecarev_sampler(sample_cnt, step=None, reduce_base=2000,
    R_range=range(37, 38), n_range=range(100, 101),
    f_range=range(11, 40), r_range=range(20, 40), seed=None):

    """
    Samples "sample_cnt" non-degenerate cellular automata instances using the given range and returns
    a sample snapshot for supervised learning task.

    Input:
        sample_cnt: Number of samples
        step: If None: a sample between 2 and no_gc value of the generated instance.
                  O.W: The step to take a snapshot (same as supervised.capture)
        seed: Only for tests to work. LEAVE AS NONE

    Example usage:
        >>> from supervised.capture import capture
        >>> from pysat.formula import CNF
        >>> sampler = ecarev_sampler(5, f_range=range(11, 12), r_range=range(61, 62), seed=41)
        >>> for sample in sampler:
        ...     print("Snapshot calculated: %i, n_gc: %i" % (sample["snapshot"], sample["no_gc"]))
        Snapshot calculated: 38, n_gc: 40
        Snapshot calculated: 2, n_gc: 21
        Snapshot calculated: 39, n_gc: 40
        Snapshot calculated: 28, n_gc: 31
        Snapshot calculated: 4, n_gc: 16

    """
    from supervised.capture import capture

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)
    log.setLevel(logging.INFO)

    random.seed(seed)
    samples_so_far = 0
    ranges = list(itertools.product(R_range, n_range, f_range, r_range))

    sample_attempts = 0.0
    while(samples_so_far < sample_cnt):
        sample_attempts += 1
        (R, n, f, r) = random.choice(ranges)
        s = random.randrange(2, pow(2, n))
        log.info("ecarev-%i-%i-%i-%i-%i" % (R, n, f, r, samples_so_far))
        res = solve(R, n, f, r, s, samples_so_far, reduce_base, save=False)

        if (not res): continue #degenerate sample
        cnf = res["cnf"]
        print("clauses %i" % len(cnf.clauses))
        no_gc = res["steps"]
        snapshot = random.randrange(2, no_gc) if (step == None) else step
        try:
            yield capture(cnf, snapshot, no_gc=no_gc, reduce_base=reduce_base)
        except Exception as err:
            log.error(err)
            traceback.print_exc()
            continue

        samples_so_far += 1

    print("success rate: %f" % (sample_cnt * 1.0 / sample_attempts))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Generates a Cellular Automata benchmark_path."""
        , epilog="Example")
    parser.add_argument('--R_min', metavar='R_min', type=int, default=0)
    parser.add_argument('--R_max', metavar='R_max', type=int, default=pow(2, 8))
    parser.add_argument('--n_min', metavar='n_min', type=int, default=100)
    parser.add_argument('--n_max', metavar='n_max', type=int, default=2000)
    parser.add_argument('--n_inc', metavar='n_inc', type=int, default=10)
    parser.add_argument('--f_min', metavar='f_min', type=int, default=20)
    parser.add_argument('--f_max', metavar='f_max', type=int, default=200)
    parser.add_argument('--f_inc', metavar='f_inc', type=int, default=1)
    parser.add_argument('--r_min', metavar='r_min', type=int, default=20)
    parser.add_argument('--r_max', metavar='r_max', type=int, default=200)
    parser.add_argument('--r_inc', metavar='r_inc', type=int, default=1)
    parser.add_argument('--max_samples', metavar='max_samples', type=int, default=2000)
    args = parser.parse_args()

    time_max = 10
    filter = GlucoseFilter(time_max = time_max)

    R_range = range(args.R_min, args.R_max + 1)
    n_range = range(args.n_min, args.n_max + 1, args.n_inc)
    f_range = range(args.f_min, args.f_max + 1, args.f_inc)
    r_range = range(args.r_min, args.r_max + 1, args.r_inc)

    sample_easy = 5
    reduce_base = 2000

    MAX_PROC = psutil.cpu_count(logical=False) - 2 # No. of physical cores (leave 2)

    res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
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

    # log.info("Ranges: R({}-{}), n({}-{}-{}), f_max({}), r_max({}), increment({})".format(
    #     args.R_min, args.R_max, args.n_min, args.n_max, args.n_inc, fwd_max, rev_max, increment))
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
                        instance = Benchmark(R, n, f, r, random.randint(1,10))

                        glucose = Glucose3(gc_freq="fixed", use_timer=True, reduce_base=reduce_base)
                        glucose.append_formula(instance.clauses)
                        glucose.time_budget(time_max)
                        res = glucose.solve_limited()

                        log.info("# of GC: R(%i)-n(%i): %i" % (R, n, glucose.nof_gc()))
                        # if (glucose.nof_gc() < 4): # degenerate
                        cnf_id = "ecarev-%i-%i-%i-%i" % (R, n, f, r)
                        if (filter.filter(glucose, cnf_id) or res == None):
                            glucose.delete()
                            continue
                        glucose.delete()

                        log.info("Candidate Found: R(%i)-n(%i)-f(%i)-r(%i)" % (R, n, f, r))

                        for cnt in range(20, sample_easy + 21):
                            s = random.randrange(2, pow(2, n))
                            job = pool.apply_async(solve, (R, n, f, r, s, reduce_base, cnt, mq))
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
