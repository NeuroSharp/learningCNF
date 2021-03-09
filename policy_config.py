from IPython.core.debugger import Tracer
import time
import logging

from settings import *
from utils import Singleton


# From MiniHyperPlane to Mini2HyperPlane, which is from [LBD] to [LBD,4]

def mhp1_2_mhp2(policy, saved_dict):
	sd = policy.state_dict()
	for k in sd:
		if k == 'policy_layers.linear_2.weight':
			sd[k][0,:]=saved_dict[k][0,:]
			sd[k][1,:]=0
			sd[k][2,:]=saved_dict[k][1,:]
		elif k == 'policy_layers.linear_2.bias':
			sd[k][0]=saved_dict[k][0]
			sd[k][1]=0
			sd[k][2]=saved_dict[k][1]
			pass
		else:
			sd[k] = saved_dict[k]
	policy.load_state_dict(sd)

# From Mini2HyperPlane to Mini3HyperPlane, which is from [LBD, 4] to [LBD,4, 1]

def mhp2_2_mhp3(policy, saved_dict):
	sd = policy.state_dict()
	for k in sd:
		if k == 'policy_layers.linear_2.weight':
			sd[k][0,:]=saved_dict[k][0,:]
			sd[k][1,:]=saved_dict[k][1,:]
			sd[k][2,:]=0
			sd[k][3,:]=saved_dict[k][2,:]
		elif k == 'policy_layers.linear_2.bias':
			sd[k][0]=saved_dict[k][0]
			sd[k][1]=saved_dict[k][1]
			sd[k][2]=0
			sd[k][3]=saved_dict[k][2]
			pass
		else:
			sd[k] = saved_dict[k]
	policy.load_state_dict(sd)
