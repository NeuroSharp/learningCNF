import re
import ipdb
from pymongo import MongoClient
from batch_model import *
from settings import *
from datautils import *
from testing import *
import numpy as np
import pickle

def test_model_from_file(model_fname, test_fname=None):
    p = re.compile('^.*run_([a-zA-Z0-9_]*)_nc([0-9]*)(.*)__([0-9]*)_epoch[0-9]+.model')
    m = p.match(model_fname)
    nc = m.group(2)
    params, dmode, config = load_hyperparams(m.group(1),int(m.group(4)))
    dmode = DataMode(dmode)
    params['data_mode'] = dmode
    settings = CnfSettings(params)
    settings.hyperparameters['num_classes'] = int(nc)

    if not test_fname:
        test_fname = config['DS_TEST_FILE']

    
    ds1 = CnfDataset(config['DS_TRAIN_FILE'],settings['threshold'],mode=settings['data_mode'])
    # ds2 = CnfDataset(test_fname, settings['threshold'], ref_dataset=ds1, mode=settings['data_mode'])    
    ds2 = CnfDataset(test_fname, 600, mode=settings['data_mode'])    
    # ds2 = CnfDataset(test_fname, 0, mode=settings['data_mode'])    
    settings.hyperparameters['max_clauses'] = ds1.max_clauses
    settings.hyperparameters['max_variables'] = ds1.max_variables    
    net = torch.load(model_fname)
    return get_embeddings(net,ds2)

def load_model_from_file(**kwargs):
    settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
    model_class = eval(settings['classifier_type'])
    net = model_class(**settings.hyperparameters)    
    return net

def load_hyperparams(name, time):
    with MongoClient() as client:
        db = client['graph_exp']
        runs = db['runs']        
        rc = runs.find_one({'experiment.name': name, 'config.hyperparams.time': time})
        g = rc['config']['data_mode'].values()
        dmode = list(list(g)[0][1].values())[0][0]
        return rc['config']['hyperparams'], dmode, rc['config']



def save_forward_embeddings(model_fname, test_fname=None, embs_fname=None):
    if not embs_fname:
        embs_fname = 'embeddings_'+model_fname
    embs, labels = test_model_from_file(model_fname,test_fname)
    with open(embs_fname,'wb+') as f:
        pickle.dump((embs, labels),f)


def nearest_k_labels(embs,labels,ind,k):
    l = labels[ind]
    a = embs - embs[ind]
    b = np.linalg.norm(a,axis=1)
    c = b.argsort()
    out = labels[c[:k]]
    # print('%d ' % l, out)
    num_correct = np.count_nonzero(out==l)    
    return num_correct / k

def get_scores_for_class_label(embs, labels, l, k):
    indices = np.where(labels==l)[0]
    res = np.asarray([nearest_k_labels(embs,labels, i, k) for i in indices])
    return res, res.mean()

