import os, sys
import subprocess
from subprocess import Popen, PIPE
import time
from contextlib import suppress
import logging
import json
import itertools
import argparse
from configparser import ConfigParser, ExtendedInterpolation
import random
import csv
import psutil
import time

current_path = os.path.dirname(os.path.abspath(__file__))
MAX_PROC = psutil.cpu_count(logical=False) - 2 # No. of physical cores (leave 2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Runs deep and vanilla glucose on a given benchmark.')
	parser.add_argument('benchmark_path', metavar='benchmark_path', type=str)
	parser.add_argument('glucose_path',   metavar='glucose_path', type=str)
	parser.add_argument('dump_path',      metavar='result_path', type=str)
	
	parser.add_argument('--run_baseline', dest='run_baseline', action='store_true', help='run both vanilla & deep Glucose')
	parser.add_argument('--no-run_baseline', dest='run_baseline', action='store_false', help='only run deep Glucose')
	parser.set_defaults(run_baseline=True)
	
	parser.add_argument('-start', metavar='start_index', dest='start', type=int)
	parser.add_argument('-end', metavar='end_index', dest='end', type=int)
	parser.set_defaults(start=0)
	parser.set_defaults(end=None)

	parser.add_argument('-slurm_node',    metavar='slurm_node', type=str)
	parser.set_defaults(slurm_node=None)
	args = parser.parse_args()

	config = ConfigParser(interpolation=ExtendedInterpolation())
	config_file = os.path.join(current_path, "deep_test.config") # TODO: Make this an argument
	config.read(config_file)
		
	benchmark_path = args.benchmark_path
	glucose_path = os.path.join(args.glucose_path, "glucose")
	dump_path = args.dump_path
	json_dump_path = os.path.join(args.dump_path, "jsons")
	with suppress(OSError):
		os.makedirs(json_dump_path)
	run_baseline = args.run_baseline
	start_ind = args.start
	end_ind   = args.end

	log = logging.getLogger(__name__)
	log.setLevel(logging.INFO)
	log_formatter  = logging.Formatter('%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.INFO)
	stream_handler.setFormatter(log_formatter)

	res_key = args.slurm_node if (args.slurm_node != None) else random.randint(10000, 99999)

	file_handler = logging.FileHandler(os.path.join(dump_path, "deep_test_{}.log".format(res_key)))
	file_handler.setFormatter(log_formatter)
	file_handler.setLevel(logging.INFO)

	log.addHandler(file_handler)
	log.addHandler(stream_handler)

	res_deep = [['file_name', 'cpu_time', 'op_cnt', 'keep_divergence', 'del_divergence',
				 'thresh_avg', 'thresh_std', 'percent_kept_rate', 'nb_ReduceDB',
				 'mem_used', 'sat_unsat']]
	res_vanilla = res_deep.copy()
	num_cols = len(res_deep[0])

	files = [os.path.join(benchmark_path, f) for f in os.listdir(benchmark_path) if os.path.isfile(os.path.join(benchmark_path, f)) and 
												f.endswith(".cnf")][start_ind:end_ind]

	tasks = []
	shared_args = [glucose_path, 
					"-jsonify", 
					"-gcFreqPolicy={}".format(config['DEFAULT']['gc_frequency_policy']),
					"-gcPolicy={}".format(config['DEFAULT']['gc_policy']),
					"-firstReduceDB={}".format(config['DEFAULT']['reduce_base']),
					"-cpu-lim={}".format(config['DEFAULT']['cpu_lim'])]
	if run_baseline:
		tasks = [shared_args + [arg, sat_file] for (sat_file, arg) in itertools.product(files, ["-use_deep_rdb", "-no-use_deep_rdb"])]
	else:
		tasks = [shared_args + ["-use_deep_rdb", sat_file] for sat_file in files]
	n_jobs = len(tasks)
	running_procs = {}

	log.info("**********************************************")
	log.info("*** Starting A Deep Test Run on: {}".format(res_key))
	log.info("*** Utilized Cores: {}".format(MAX_PROC))
	log.info("*** Benchmark : {} (size: {})".format(benchmark_path, len(tasks)))
	log.info("*** Baseline  : {}".format("Yes" if run_baseline else "No"))
	log.info("*** Dump Path : {}".format(dump_path))
	log.info("*** -------------- Config Params -------------")
	for section in config.sections() + ["DEFAULT"]:
		for (config_key, config_val) in config.items(section):
			log.info("*** {}.{} : {}".format(section, config_key, config_val))
	log.info("**********************************************")
	log.info("")

	def handle_results(proc):
		result = ""
		timeout = False
		file_path = proc.args[-1]
		file_name = os.path.basename(file_path)

		if os.path.exists("{}.json".format(file_path)):
			json_file = open("{}.json".format(file_path))
			stats = json.load(json_file)

			result = [file_name, stats['cpu_time'], stats['op_cnt'], stats['keep_divergence'], 
					  stats['del_divergence'], stats['thresh_avg'], stats['thresh_std'], 
					  stats['percent_kept_rate'], stats['nb_ReduceDB'], stats['mem_used'],
					  stats['result']]
			
			log.info("[Finished] {}, cpu_time:{}s".format(" ".join(proc.args), stats['cpu_time']))
		else:
			timeout = True
			result = [""] * num_cols
			result[0]  = file_name
			result[1]  = config['DEFAULT']['cpu_lim']
			result[-1] = "timeout"

			log.info("[Timeout] {}".format(" ".join(proc.args)))

		if proc.args.count("-use_deep_rdb") > 0:
			res_deep.append(result)

			if not timeout:
				os.rename(json_file.name, os.path.join(json_dump_path, "{}-deep.json".format(file_name)))
		else:
			res_vanilla.append(result)
			if not timeout:
				os.rename(json_file.name, os.path.join(json_dump_path, "{}-vanilla.json".format(file_name)))

	def flush_results(tc):
		file = open(os.path.join(dump_path, 'deep_res_{}.csv'.format(res_key)), 'w+')
		writer = csv.writer(file)
		writer.writerows(res_deep)
		file.close()

		file = open(os.path.join(dump_path, 'vanilla_res_{}.csv'.format(res_key)), 'w+')
		writer = csv.writer(file)
		writer.writerows(res_vanilla)
		file.close()

		log.info("Dumped {}/{} results to file".format(tc, n_jobs))


	task_count = 0
	completed_task_count = 0

	while tasks or running_procs:
		while len(running_procs) < MAX_PROC and tasks:
			task = tasks.pop()
			task_count += 1
			log.info("Running {}/{}: {}".format(task_count, n_jobs, " ".join(task)))
			process = Popen(task)
			running_procs[process.pid] = process

		for pid in list(running_procs.keys()):
			retcode = running_procs[pid].poll()
			if retcode is not None: # Process finished.
				proc = running_procs.pop(pid)
				try:
					handle_results(proc)
				except Exception as e:
					log.info("Exception while processing: {}".format(proc.args[-1]))

				completed_task_count += 1

				if completed_task_count % MAX_PROC == 0 : 
					flush_results(completed_task_count)
		time.sleep(10)

	flush_results(completed_task_count)

	log.info("Task on {} Finished Successfully.".format(res_key))