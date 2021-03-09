import argparse
from configparser import ConfigParser, ExtendedInterpolation
import subprocess
from subprocess import Popen, PIPE
import os
import math
import logging
from contextlib import suppress
import shutil
import glob
import pandas as pd
import signal
import pickle
import torch
import time

current_path = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../.."))


def run_in_order(task_list):
	for task in task_list:
		process = Popen(task[2:], shell=task[1], stdout=PIPE, stderr=PIPE, env=env)
		
		log.info("{} (pid: {}):".format(task[0], process.pid))
		log.info("\t>>> {}".format(" ".join(task[2:])))
		
		result = process.communicate()
		if (process.poll() != 0):
			log.error("[{}] task failed:\n >>> {}".format(task[0], result[1].decode('UTF-8')))
			exit(1)

def run_in_parallel(task_list):
	running_procs = {}
	for task in task_list:
		process = Popen(task[3:], shell=task[1], stdout=PIPE, stderr=PIPE, env=env)
		running_procs[process.pid] = (task[2], process) # (Node, Process)

		log.info("{} (pid on scheduler: {}):".format(task[0], process.pid))
		log.info("\t>>> {}".format(" ".join(task[3:])))
	
	while (len(running_procs) > 0):
		time.sleep(60)
		for pid in list(running_procs.keys()):
			retcode = running_procs[pid][1].poll()
			if retcode is not None: # Process finished.
				(node, proc) = running_procs.pop(pid)
				if (retcode == 0):
					log.info("Task on {} finished. (pid on scheduler: {})".format(node, pid))
				else:
					log.error("Task on {} failed. (pid on scheduler: {})".format(node, pid))
					log.error(proc.communicate()[1])
					

def terminate(signum, frame):
	log.info("Received signal [{}].".format(signum))

	nodelist = args.slurm_list
	if(len(nodelist) > 0):
		tasks = [
			['Canceling Jobs', False, 'scancel', '-w'].extend(nodelist)
		]
		run_in_order(tasks)

	exit(0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""Takes a solver model and compiles a deep-glucose
		solver from it. Then it runs deep and vanilla versions on a given benchmark on SLURM.""",
		epilog="Example")
	parser.add_argument('exp_name'      , metavar='experiment_name', type=str)
	parser.add_argument('model_path'    , metavar='model', type=str)
	parser.add_argument('benchmark_path', metavar='benchmark', type=str)
	parser.add_argument('slurm_list'    , metavar='slurm_node_list', nargs='+', type=str)
	args = parser.parse_args()

	config = ConfigParser(interpolation=ExtendedInterpolation())
	config_file = os.path.join(current_path, "defaults.config") # TODO: Make this an argument
	config.read(config_file)
	config.set('DEFAULT', 'root', root)

	env = os.environ.copy()
	env["MROOT"] = config["GLUCOSE"]["path"]

	model_name = os.path.splitext(os.path.basename(args.model_path))[0]
	dump_path = os.path.expanduser(os.path.join(config["DEEP_TEST"]["dump_path"], args.exp_name)) + "/"
	with suppress(OSError):
		os.mkdir(dump_path)
	shutil.copy(args.model_path, dump_path)
	bench_size = len([f for f in os.listdir(args.benchmark_path) if os.path.isfile(os.path.join(args.benchmark_path, f)) and 
					 							f.endswith(".cnf")])

	git_commit_id = subprocess.check_output(["git", "-C", root, "describe", "--always"]).strip().decode('UTF-8')
	git_branch = subprocess.check_output(["git", "-C", root, "branch"]).strip().decode('UTF-8')

	log = logging.getLogger(__name__)
	log.setLevel(logging.INFO)
	log_formatter  = logging.Formatter('[%(levelname)s %(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(log_formatter)
	log.addHandler(stream_handler)
	
	file_handler = logging.FileHandler(os.path.join(dump_path, "slurm_runner.log"))
	file_handler.setFormatter(log_formatter)
	file_handler.setLevel(logging.INFO)

	log.addHandler(file_handler)
	log.addHandler(stream_handler)

	signal.signal(signal.SIGINT,  terminate)
	signal.signal(signal.SIGTERM, terminate)

	log.info("**********************************************")
	log.info("*** Starting SLURM Runner on: {}".format(", ".join(args.slurm_list)))
	log.info("*** Experiment Name: {}".format(args.exp_name))
	log.info("*** Model Name: {}".format(model_name))
	log.info("*** Benchmark : {} (size: {})".format(args.benchmark_path, bench_size))
	log.info("*** Dump Path : {}".format(dump_path))
	log.info("*** Commit-id : {} on {} branch".format(git_commit_id, git_branch))
	log.info("*** -------------- Config Params -------------")
	for section in config.sections() + ["DEFAULT"]:
		for (config_key, config_val) in config.items(section):
			log.info("*** {}.{} : {}".format(section, config_key, config_val))
	log.info("**********************************************")
	log.info("")

	log.info("Extracting Model (torch -> numpy).")
	numpy_model_path = os.path.join(dump_path, "{}.pickle".format(model_name))
	torch_model = torch.load(args.model_path)
	np_model = {}
	for k in torch_model.keys():
		np_model[k] = torch_model[k].numpy()
	with open(numpy_model_path,'wb') as f:
		pickle.dump(np_model,f)

	tasks = [
		['Embedding Model', False, 'python', os.path.join(config["GLUCOSE"]["path"], "model/compiler.py"), numpy_model_path, config["GLUCOSE"]["template_path"], "--clean"],
		['Compile Glucose', False, 'make', '-C', os.path.join(config["GLUCOSE"]["path"],"core"), "clean", "r"]
	]
	run_in_order(tasks)

	# copy binary glucose to th dump folder
	glucose_bin = os.path.join(dump_path, "glucose")
	shutil.copy(os.path.join(config["GLUCOSE"]["path"],"core", "glucose_release"), glucose_bin)

	# setup the slurm jobs
	slurm = config["SLURM"]
	baseline = "--no-run_baseline" if (config["DEEP_TEST"]["run_baseline"].lower() == "false") else "--run_baseline"
	mail_type = config["SLURM"]["mail-type"] or 'None'
	mail_user = config["SLURM"]["mail-user"] or 'None'
	chunk_size = math.ceil(bench_size / len(args.slurm_list))
	tasks = [
		['Srun_%d' % i, False, nodelist, 'srun', '--partition=%s' % slurm["partition"], '--nodelist=%s' % nodelist, 
			'--mail-type=%s' % mail_type, '--mail-user=%s' % mail_user,
			'--pty', 'python', os.path.join(current_path, "deep_test.py"), args.benchmark_path, dump_path,
			dump_path, baseline, "-slurm_node", nodelist,
			"-start", str(i * chunk_size), "-end", str(min((i+1) * chunk_size, bench_size))
		] for (i, nodelist) in enumerate(args.slurm_list)
	]
	run_in_parallel(tasks)

	# combine the result csv files from all the nodes
	log.info("Combining the results from multiple nodes.")
	for prefix in ["deep", "vanilla"]:
		res = glob.glob(os.path.join(dump_path, "{}_res_cpunode*.csv".format(prefix)))
		#combine all files in the list
		combined_csv = pd.concat([pd.read_csv(f) for f in res])
		#export to csv
		combined_csv.to_csv(os.path.join(dump_path, "{}_res.csv".format(prefix)), index=False, encoding='utf-8-sig')

	log.info("Slurm_runner.py finished successfully.")
	

