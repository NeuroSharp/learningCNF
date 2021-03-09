import os
import ipdb
import random
import string
import ray
import logging
import argparse
import pickle
import itertools
import numpy as np

from os import listdir
from pysat.formula import CNF
from pysat._fileio import FileObject

def random_string(n):
  return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(n)])

@ray.remote
def generate_datum(fname, dest, step):
	from supervised.capture import capture
	# with FileObject(fname, mode='r', compression='use_ext') as fobj:
	# 	formula_str = fobj.fp.read()
	# 	formula = CNF(from_string=formula_str)

	print('generate_darum called with {}'.format(fname))
	cnf = CNF(from_file=fname)
	try:
		rc = capture(cnf,step)
		print('capture finished')
		with open('{}/{}_step_{}.pickle'.format(dest,os.path.basename(fname),step),'wb') as f:
			pickle.dump(rc,f)
	except Exception as e:
		print('capture threw exception')
		pass
	return True

@ray.remote
def generate_from_sampler(dest, step=None, prefix='sample', reduce_base=1000, R_range=range(37, 38), n_range=range(50, 51),
    f_range=range(11, 20), r_range=range(11, 20),num_samples=1):

	from ecarev.glucose_gen import ecarev_sampler
	fname_prefix = '{}_{}-{}_{}-{}_{}-{}_{}-{}_{}'.format(prefix,n_range.start, n_range.stop, R_range.start, R_range.stop,
		f_range.start, f_range.stop, r_range.start, r_range.stop, reduce_base)
	try:
		sampler = ecarev_sampler(num_samples, step=step, reduce_base=reduce_base, R_range=R_range, n_range=n_range,
		f_range=f_range, r_range=r_range)
		for sample in sampler:
			fname = '{}_{}'.format(fname_prefix,random_string(8))
			print('generated {}'.format(fname))
			with open('{}/{}.pickle'.format(dest,fname),'wb') as f:
				pickle.dump(sample,f)
	except Exception as e:
		print('capture threw exception')
		print(e)
		pass
	return True

def load_dir(directory):
  return load_files([os.path.join(directory, f) for f in listdir(directory)])

def load_files(files):
  if type(files) is not list:
    files = [files]
  only_files = [x for x in files if os.path.isfile(x)]
  only_dirs = [x for x in files if os.path.isdir(x)]
  return only_files if not only_dirs else only_files + list(itertools.chain.from_iterable([load_dir(x) for x in only_dirs]))

def generate_online(args):
	assert args.n, "Must define target number"
	dst = args.destination_dir
	try:
		os.mkdir(dst)
	except:
		pass
	results = [generate_from_sampler.remote(dst) for _ in range(args.n)]
	vals = ray.get(results)
	print('Finished')

def generate_dataset(args):
	src = load_dir(args.source_dir)
	dst = args.destination_dir
	try:
		os.mkdir(dst)
	except:
		pass
	results = []
	np.random.shuffle(src)
	for i, fname in enumerate(src):
		results.append(generate_datum.remote(fname,dst,random.choice([-1,0,1])+args.step))
		if args.n >0 and i>=args.n:
			break

	vals = ray.get(results)
	print('Finished')


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Process some params.')
	# parser.add_argument('params', metavar='N', type=str, nargs='*', help='an integer for the accumulator')
	parser.add_argument('-s', '--source_dir', type=str, help='source directory')
	parser.add_argument('-d', '--destination_dir', type=str, help='destination directory')
	parser.add_argument('-n', type=int, default=0, help='hard cap on number of formulas')
	parser.add_argument('-p', '--parallelism', type=int, default=1, help='number of cores to use (Only if not in cluster mode)')
	parser.add_argument('-t', '--step', type=int, default=2, help='step to capture')
	parser.add_argument('-c', '--cluster', action='store_true', default=False, help='run in cluster mode')
	parser.add_argument('-o', '--online', action='store_true', default=False, help='Create ecarev CNF on the fly')
	# parser.add_argument('-c', '--cluster', action='store_true', default=False, help='settings file')
	args = parser.parse_args()
	if args.cluster:
		ray.init(address='auto', redis_password='blabla')
	else:
		ray.init(num_cpus=args.parallelism)

	if args.online:
		generate_online(args)
	else:
		generate_dataset(args)
