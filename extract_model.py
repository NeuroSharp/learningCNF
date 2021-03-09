import os
import pickle
import torch
import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract a torch model into numpy')
    parser.add_argument('params', metavar='1', type=str, nargs='*',
                        help='torch_model')
    parser.add_argument('-o', '--out', type=str, default=None, help='Output file name') 
    args = parser.parse_args()

    assert(len(args.params)>0)
    modelname = args.params[0]
    if args.out:
    	outfile = args.out
    else:
    	outfile = os.path.basename(modelname)+'.pickle'

    a = torch.load(modelname)
    z = {}
    for k in a.keys():
    	z[k] = a[k].numpy()
    with open(outfile,'wb') as f:
    	pickle.dump(z,f)

    print('Wrote {}'.format(outfile))

if __name__=='__main__':
    main()
