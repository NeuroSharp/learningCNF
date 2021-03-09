import pickle, os, sys
import logging
import numpy as np
import argparse
from contextlib import suppress

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Runs deep and vanilla glucose on a given benchmark.')
	parser.add_argument('model_path', metavar='model_path', type=str)
	parser.add_argument('template_prefix', metavar='template_prefix', type=str)
	parser.add_argument('--clean', dest='clean', action='store_true', help='clean build')
	parser.set_defaults(clean=False)
	args = parser.parse_args()

	log = logging.getLogger(__name__)
	log.setLevel(logging.INFO)
	log_formatter  = logging.Formatter('%(levelname)s %(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(log_formatter)
	log.addHandler(stream_handler)

	current_path = os.path.dirname(os.path.abspath(__file__))

	if (args.clean):
		with suppress(OSError):
			os.remove(os.path.join(current_path, "Model.h"))
			os.remove(os.path.join(current_path, "Model.cc"))

	MODEL_F = args.model_path
	TEMPLATE_h_F = "{}.h.template".format(args.template_prefix)
	TEMPLATE_cc_F = "{}.cc.template".format(args.template_prefix)

	templatef = os.path.abspath(TEMPLATE_h_F)
	f = open(templatef, 'r') #, encoding = "ISO-8859-1")
	template_h = f.read()

	templatef = os.path.abspath(TEMPLATE_cc_F)
	f = open(templatef, 'r') #, encoding = "ISO-8859-1")
	template_c = f.read()

	modelf = os.path.abspath(MODEL_F)
	try:
	    with open(modelf, 'rb') as f :
	        model = pickle.load(f)
	        for key in model.keys():
	        	if len(model[key]) == 1:
	        		model[key] = model[key].flatten().tolist()

	except FileNotFoundError:
		log.error("'{}' file not found.".format(modelf))
		exit(1)

	values = {}
	for key in model.keys():
		if key == 'state_vbn.moving_mean': continue
		# make the keys c++ compatible
		new_key = key.replace(".", "_")
		
		# handle 2-d arrays
		if len(np.shape(model[key])) == 2:
			model_T = model[key].T # Transposing because the input models have wronge shape
			values[new_key] = ", ".join([ '{' + ", ". join(row) + '}' for row in model_T.astype(np.str).tolist()])
			values["{}_shape_0".format(new_key)] = len(model_T)
			values["{}_shape_1".format(new_key)] = len(model_T[0])
		else:
			values[new_key] = ', '.join(map(str, model[key]))
			values["{}_shape_0".format(new_key)] = len(model[key])
		
	values['model_file_name'] = modelf
	
	f = open(os.path.join(current_path, "Model.h"), 'w')
	f.write(template_h.format(**values))

	f = open(os.path.join(current_path, "Model.cc"), 'w')
	f.write(template_c.format(**values))
