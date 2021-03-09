import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from batch_model import *
from datautils import *
import re
import utils
import time
import numpy as np
import ipdb
from enum import Enum
from tensorboard_logger import configure, log_value
from sacred import Experiment
from testing import test

EX_NAME = 'trenery_4'


class BaseMode(Enum):
    ALL = 1
    EMBEDDING = 2


ex = Experiment(EX_NAME)

# torch.manual_seed(1)

# TRAIN_FILE = 'expressions-synthetic/boolean5.json'

TRAIN_FILE = 'expressions-synthetic/split/boolean5-trainset.json'
VALIDATION_FILE = 'expressions-synthetic/split/boolean5-validationset.json'
TEST_FILE = 'expressions-synthetic/split/boolean5-testset.json'

DS_TRAIN_TEMPLATE = 'expressions-synthetic/split/%s-trainset.json'
DS_VALIDATION_TEMPLATE = 'expressions-synthetic/split/%s-validationset.json'
DS_TEST_TEMPLATE = 'expressions-synthetic/split/%s-testset.json'

PRINT_LOSS_EVERY = 100
# PRINT_LOSS_EVERY = 20
VALIDATE_EVERY = 1000
NUM_EPOCHS = 400
# NUM_EPOCHS = 150
LOG_EVERY = 10
# SAVE_EVERY = 1
SAVE_EVERY = 5


ACC_LR_THRESHOLD = 0.02

# a^b -> c
# 1 -3
# 2 -3
# -1 -2 3

# train_formula = [[[1,-3],[-1,-2,3]],
#                  [[2,-3],[-1,-2,3]], 
#                  [[1,-3],[2,-3],[-1,-2,3]]]



# hyperparams = {
#     'embedding_dim': 4,
#     'max_clauses': 3, 
#     'max_variables': 3, 
#     'num_ground_variables': 3, 
#     'dataset': 'boolean8',
#     'model_dir': 'saved_models',
#     'max_iters': 6,
#     'batch_size': 4,
#     'val_size': 100, 
#     # 'classifier_type': 'GraphLevelClassifier',
#     'classifier_type': 'EqClassifier',
#     'combinator_type': 'SimpleCombinator',
#     'use_ground': False,
#     'split': False,
#     'cuda': False
# }

@ex.capture
def log_name(settings):
    name = ex.current_run.experiment_info['name']
    return 'run_%s_nc%d_bs%d_ed%d_iters%d__%s' % (name, settings['num_classes'], 
        settings['batch_size'], settings['embedding_dim'], 
        settings['max_iters'], settings['exp_time'])
    
# @ex.capture
def train(ds, ds_validate=None, net=None):
    settings = CnfSettings()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, len(ds))
    # trainloader = torch.utils.data.DataLoader(ds, batch_size=settings['batch_size'], sampler = sampler, pin_memory=settings['cuda'], collate_fn = cnf_collate)
    trainloader = torch.utils.data.DataLoader(ds, batch_size=settings['batch_size'], sampler = sampler, collate_fn = cnf_collate)
    print('%d classes, %d samples'% (ds.num_classes,len(ds)))
    settings.hyperparameters['num_classes'] = ds.num_classes
    if not 'max_clauses' in settings.hyperparameters:
        settings.hyperparameters['max_clauses'] = ds.max_clauses
    settings.hyperparameters['max_variables'] = ds.max_variables

    current_time = time.time()
    cl_type = eval(settings['classifier_type'])
    base_model = settings['base_model']
    net = cl_type(**(settings.hyperparameters))
    if base_model:
        base_mode = settings['base_mode']
        net.load_state_dict(torch.load(base_model))        
        if base_mode == BaseMode.EMBEDDING:
            encoder = net.encoder
            embedder = net.embedder
            embedder.settings = settings
            encoder.settings = settings
            net = cl_type(encoder=encoder, embedder=embedder, **(settings.hyperparameters))            
        # p = re.compile('^.*run_.*_epoch([0-9]+).model')
        # m = p.match(base_model)
        # start_epoch = int(m.group(1))
    else:
        net = cl_type(**(settings.hyperparameters))

    if settings.hyperparameters['cuda']:
        net.cuda()
    else:
        net.cpu()

    start_epoch = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=settings['init_lr'], momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=settings['init_lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    batch_size = settings['batch_size']
    get_step = lambda x,y: x*len(trainloader)+y
    configure("runs/%s" % log_name(settings), flush_secs=5)
    oldnet = []

    total_correct = 0
    last_v_acc = 0
    for epoch in range(start_epoch,NUM_EPOCHS):
        running_loss = 0.0
        utils.exp_lr_scheduler(optimizer, epoch, init_lr=settings['init_lr'], lr_decay_epoch=settings['decay_num_epochs'],decay_rate=settings['decay_lr'])
        for i, data in enumerate(trainloader, 0):
            inputs = Variable(data['variables'], requires_grad=False)
            effective_bs = len(inputs)
            if  effective_bs != settings['batch_size']:
                print('Trainer gave us shorter batch!!')
                # continue
            topvar = torch.abs(Variable(data['topvar'], requires_grad=False))
            labels = Variable(data['label'], requires_grad=False)
            cmat_pos = Variable(data['sp_v2c_pos'], requires_grad=False)
            cmat_neg = Variable(data['sp_v2c_neg'], requires_grad=False)
            ipdb.set_trace()
            ind = data['idx_in_dataset']
            if settings.hyperparameters['cuda']:
                topvar, labels = topvar.cuda(), labels.cuda()
                inputs, cmat_pos, cmat_neg = inputs.cuda(), cmat_pos.cuda(), cmat_neg.cuda()

            # zero the parameter gradients
            
            optimizer.zero_grad()
            # print('iteration %d beginning...' % i)
            # forward + backward + optimize
            outputs, aux_losses = net(inputs, output_ind=topvar, cmat_pos=cmat_pos, cmat_neg=cmat_neg, batch_size=effective_bs)
            loss = criterion(outputs, labels)   # + torch.sum(aux_losses)
            try:
                loss.backward()
            except RuntimeError as e:
                print('Woah, something is going on')
                print(e)
                # ipdb.set_trace()            
            

            if settings['moving_ground'] and (net.encoder.embedding.weight.grad != net.encoder.embedding.weight.grad).data.any():
                print('NaN in embedding grad!!!')
                pdb.set_trace()

            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

            correct = outputs.max(dim=1)[1]==labels
            num_correct = torch.nonzero(correct.data).size()
            if len(num_correct):
                total_correct += num_correct[0]
            if get_step(epoch,i) % LOG_EVERY == 0:
                log_value('loss',loss.data[0],get_step(epoch,i))
            if i % PRINT_LOSS_EVERY == PRINT_LOSS_EVERY-1:                
                new_time = time.time()                
                print('Average time per mini-batch, %f' % ((new_time-current_time) / PRINT_LOSS_EVERY))
                current_time = new_time
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / PRINT_LOSS_EVERY))
                running_loss = 0.0
                print('[%d, %5d] training accuracy: %.3f' %
                      (epoch + 1, i + 1, total_correct / (PRINT_LOSS_EVERY*settings['batch_size'])))
                total_correct = 0.0   

            # if ds_validate and i>0 and i % VALIDATE_EVERY == 0:

        # Validate and recompute learning rate
        if ds_validate is not None:
            v_loss, v_acc = test(net, ds_validate, weighted_test=True)
            v_loss = v_loss.data.numpy() if not settings['cuda'] else v_loss.cpu().data.numpy()
            print('Validation loss %f, accuracy %f' % (v_loss,v_acc))
            log_value('validation_loss',v_loss,get_step(epoch,i))
            log_value('validation_accuracy',v_acc,get_step(epoch,i))
            # scheduler.step(v_loss)


        # if epoch>0 and epoch % SAVE_EVERY == 0:            
        if epoch % SAVE_EVERY == 0:            
            torch.save(net.state_dict(),'%s/%s_epoch%d.model' % (settings['model_dir'],log_name(settings), epoch))
            if settings['reset_on_save']:
                print('Recreating model!')
                print(settings.hyperparameters)
                oldnet.append(net)
                net = cl_type(**(settings.hyperparameters))
                if settings.hyperparameters['cuda']:
                    net.cuda()                    
                net.load_state_dict(torch.load('%s/%s_epoch%d.model' % (settings['model_dir'],log_name(settings), epoch)))
                pdb.set_trace()


    print('Finished Training')
