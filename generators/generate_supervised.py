import os
import sys
import random
import string
import ray
import logging
import argparse
import itertools
import numpy as np
from contextlib import suppress

from pysat.formula import CNF
from pysat._fileio import FileObject

from supervised.op_count_capture import *

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s')
log = logging.getLogger(__name__)


@ray.remote
def generate_datum(fname, config):
    log.info(f"Sampling {config['max_samples_per_formula']} samples from {fname}...")

    captr = OpCountCapture(fname, **config)
    sample_cnt = 0
    stats = []
    while (sample_cnt < config['max_samples_per_formula']):
        try:
            sample_cnt += 1
            stats += [captr.capture()]
            captr.dump(config['dir'])
        except Exception as e:
            log.error(e, exc_info=True)
            continue

    return stats

def load_dir(directory):
    return load_files([os.path.join(directory, f) for f in os.listdir(directory)])

def load_files(files):
    if type(files) is not list:
        files = [files]
    only_files = [x for x in files if os.path.isfile(x)]
    only_dirs = [x for x in files if os.path.isdir(x)]
    return only_files if not only_dirs else only_files + list(itertools.chain.from_iterable([load_dir(x) for x in only_dirs]))

def generate_dataset(args):
    src = load_dir(args.source_dir)
    dst = args.destination_dir

    config = {
        'dir': dst,
        'max_samples_per_formula': args.max_samples_per_formula,
    }
    for p in args.params:
        k, v = p.split('=')
        config[k]=v
    with suppress(OSError):
        os.mkdir(dst)

    stats = []
    np.random.shuffle(src)
    for i, fname in enumerate(src):
        stats.append(generate_datum.remote(fname, config))
        if args.n >0 and i>=args.n:
            break

    vals = [y for x in ray.get(stats) for y in x] # Flatten the results
    print(vals)
    if (args.viz):
        OpCountCapture.dump_stats(vals, dst)

    print('Finished')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some params.')
    parser.add_argument('params', metavar='N', type=str, nargs='*', help='capturer specific params.')
    parser.add_argument('-s', '--source_dir', type=str, help='source directory')
    parser.add_argument('-d', '--destination_dir', type=str, help='destination directory')
    parser.add_argument('-n', type=int, default=0, help='hard cap on number of input formulas to examine')
    parser.add_argument('-p', '--parallelism', type=int, default=1, help='number of cores to use (Only if not in cluster mode)')
    parser.add_argument('-c', '--cluster', action='store_true', default=False, help='run in cluster mode')
    parser.add_argument('-m', '--max_samples_per_formula', type=int, help='max number of samples taken from each input formula.')
    parser.add_argument('-v', '--viz', action='store_true', default=False, help='visualize the satas in a plot')

    args = parser.parse_args()
    if args.cluster:
        ray.init(address='auto', redis_password='blabla')
    else:
        ray.init(num_cpus=args.parallelism)

    generate_dataset(args)
