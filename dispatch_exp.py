import argparse
import functools
from IPython.core.debugger import Tracer
import subprocess
import json
import asyncio
import signal
import itertools

from dispatch_utils import *
from gridparams import *

NAME = 'DEBUG_TEST'
LOCAL_CMD = ['python', 'run_exp.py']
MONGO_MACHINE = 'russell'
MONGO_SFX = ':27017:'
EXP_CNF_SFX = 'graph_exp'
EXP_QBF_SFX = 'qbf_exp'
EXP_RL_SFX = 'rl_exp'

dop_machines = ['wessel', 'russell']
all_machines = []

loop = asyncio.get_event_loop()

def make_safe(f, blacklist):
	def wrapper(x):
		if x in blacklist:
			return x
		else:
			return f(x)
	return wrapper

safe_remove_machine = make_safe(remove_machine,dop_machines)
safe_async_remove_machine = make_safe(async_remove_machine,dop_machines)

def cleanup_handler(signame):
	print('Got ctrl-c, cleaning up synchronously')
	loop.run_until_complete(loop.shutdown_asyncgens())
	loop.stop()
	loop.close()
	for mname in all_machines:
		safe_remove_machine(mname)

	exit()

def main():
	parser = argparse.ArgumentParser(description='Process some params.')
	parser.add_argument('params', metavar='N', type=str, nargs='*',
	                    help='Experiment parameters')
	parser.add_argument('--name', type=str, help='Experiment name')
	parser.add_argument('-f', '--file', type=str, help='Settings file')	
	parser.add_argument('-c', '--command', type=str, default='reinforce_exp.py', help='Command to run (eg: qbf_exp.py)')	
	parser.add_argument('-t', '--instance-type', type=str, help='instance type (eg: t2.xlarge)')	
	parser.add_argument('-m', '--machine', type=str, help='machine name (eg: exp_dqn)')	
	parser.add_argument('--commit', type=str, default='rl', help='commit to load')	
	parser.add_argument('-n', '--num', type=int, default=1, help='Number of concurrent experiments')	
	parser.add_argument('--rm', action='store_true', default=False, help='Delete after experiment is done')	
	args = parser.parse_args()

	if args.name is None:
		print('Name is NOT optional')
		exit()
	base_mname = args.machine if args.machine else machine_name(args.name)
	params = args.params

# override params, cmdargs > params file > params defined in source code.

	grid = GridParams(args.file)
	def_params = grid.grid_dict()
	for k in params:
		a, b = k.split('=')
		b = b.split(',')
		def_params[a]=b
	configs = list(utils.dict_product(def_params))

	mongo_addr = get_mongo_addr(MONGO_MACHINE)+MONGO_SFX
	if args.command == 'run_exp.py':
		mongo_addr += EXP_CNF_SFX
	elif args.command == 'qbf_exp.py':
		mongo_addr += EXP_QBF_SFX
	elif args.command == 'reinforce_exp.py':
		mongo_addr += EXP_RL_SFX
	else: 
		mongo_addr += 'unknown'

	# Tracer()()

	all_executions = []
	all_experiments = list(itertools.product(range(args.num),range(len(configs))))
	for i, conf_num in all_experiments:		
		conf = configs[conf_num]
		a = ['%s=%s' % j for j in conf.items()]	
		if conf:
			a.insert(0, 'with')
		a.insert(0, '--name %s-%d-%d' % (str(args.name),i,conf_num))
		a.insert(0, '-m %s' % mongo_addr)
		a.insert(0, '%s' % args.command)
		a.insert(0, 'python')

		mname = args.machine if args.machine else base_mname+'-{}'.format(i)
		p = async_dispatch_chain(mname,a, args.instance_type, args.rm, args.commit)
		all_executions.append(p)

	for signame in ('SIGINT', 'SIGTERM'):
		loop.add_signal_handler(getattr(signal, signame), functools.partial(cleanup_handler, signame))
	loop.run_until_complete(asyncio.gather(*all_executions))
	loop.close()
	# else:
	# 	execute_machine(mname," ".join(a))
	print('Experiment finished')
	# if args.rm:
	# 	print('Removing machine %s ...' % mname)
	# 	remove_machine(mname)


async def async_dispatch_chain(mname, params, instance_type, rm, commit_name):
	all_machines.append(mname)
	if not machine_exists(mname):
		print('Provisioning machine %s...' % mname)
		rc = await async_provision_machine(mname,instance_type,commit_name)
	else:
		print('Machine already exists, setting up...')
		rc = await async_setup_machine(mname,commit_name)
	print('Running experiment %s on machine %s...' % (params[0],mname))
	p = await async_execute_machine(mname," ".join(params))
	print('Experiment {} finished!'.format(mname))
	if rm and mname not in dop_machines:
		print('Removing machine %s ...' % mname)
		await safe_async_remove_machine(mname)
		print('Removed machine %s ...' % mname)

if __name__=='__main__':
	main()
