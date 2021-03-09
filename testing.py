import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.nn.modules.distance import CosineSimilarity
from datautils import *
from settings import *
import utils
import pdb

def test(model, ds: CnfDataset, **kwargs):
    settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
    test_bs = 10
    # test_bs = settings['batch_size']
    c_size = settings['max_clauses']
    v_size = settings['max_variables']
    criterion = nn.CrossEntropyLoss()
    if 'weighted_test' in kwargs and kwargs['weighted_test']:
        sampler = torch.utils.data.sampler.WeightedRandomSampler(ds.weights_vector, len(ds))
        vloader = torch.utils.data.DataLoader(ds, batch_size=test_bs, sampler = sampler, collate_fn = cnf_collate)
        # vloader = torch.utils.data.DataLoader(ds, batch_size=test_bs, sampler = sampler, pin_memory=settings['cuda'], collate_fn = cnf_collate)
    else:
        sampler = torch.utils.data.sampler.RandomSampler(ds)
        vloader = torch.utils.data.DataLoader(ds, batch_size=test_bs, sampler = sampler, collate_fn = cnf_collate)
    total_loss = 0
    total_correct = 0
    total_iters = 0
    print('Begin testing, number of mini-batches is %d' % len(vloader))

    for _,data in zip(range(settings['val_size']),vloader):
        inputs = Variable(data['variables'], requires_grad=False)
        cmat_pos = Variable(data['sp_v2c_pos'], requires_grad=False)
        cmat_neg = Variable(data['sp_v2c_neg'], requires_grad=False)

        # pad inputs
        s = inputs.size()
        padsize = settings['max_variables'] - ds.max_variables
        if s and padsize > 0:
            if settings['sparse']:
                cmat_pos = Variable(torch.sparse.FloatTensor(cmat_pos.data._indices(),cmat_pos.data._values(),torch.Size([c_size*test_bs,v_size*test_bs])))
                cmat_neg = Variable(torch.sparse.FloatTensor(cmat_neg.data._indices(),cmat_neg.data._values(),torch.Size([c_size*test_bs,v_size*test_bs])))
            else:
                pad = Variable(settings.zeros([s[0],padsize,s[2]]),requires_grad=False)
                inputs = torch.cat([inputs,pad.double()],1)

        if  len(inputs) != test_bs:
            print('Trainer gave us no batch!!')
            continue
        topvar = torch.abs(Variable(data['topvar'], requires_grad=False))
        labels = Variable(data['label'], requires_grad=False)
        if settings.hyperparameters['cuda']:
            topvar, labels = topvar.cuda(), labels.cuda()
            inputs, cmat_pos, cmat_neg = inputs.cuda(), cmat_pos.cuda(), cmat_neg.cuda()
        outputs, aux_losses = model(inputs, output_ind=topvar, batch_size=test_bs, cmat_pos=cmat_pos, cmat_neg=cmat_neg)
        loss = criterion(outputs, labels)   # + torch.sum(aux_losses)
        correct = outputs.max(dim=1)[1]==labels
        num_correct = torch.nonzero(correct.data).size()
        if len(num_correct):
            total_correct += num_correct[0]
        total_loss += loss
        total_iters += 1
        print('Testing, iteration %d, total_correct = %d' % (total_iters, total_correct))

    return total_loss, total_correct / (total_iters*test_bs)


def siamese_test(model, ds, **kwargs):
    test_bs = 5
    dist = CosineSimilarity(dim=1)
    settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()
    criterion = nn.CosineEmbeddingLoss(margin=settings['cosine_margin'])
    sampler = torch.utils.data.sampler.RandomSampler(ds)
    vloader = torch.utils.data.DataLoader(ds, batch_size=test_bs, sampler = sampler)
    total_loss = 0
    total_correct = 0
    total_iters = 0
    for _,data in zip(range(settings['val_size']),vloader):
        inputs = (data['left']['variables'],data['right']['variables'])
        if  len(inputs[0]) != test_bs:
            print('loader gave us no batch!!')
            continue
        topvar = (torch.abs(Variable(data['left']['topvar'], requires_grad=False)), torch.abs(Variable(data['right']['topvar'], requires_grad=False)))
        labels = Variable(data['label'], requires_grad=False)
        if settings.hyperparameters['cuda']:
            labels = labels.cuda()
            topvar = (topvar[0].cuda(), topvar[1].cuda())
            inputs = [t.cuda() for t in inputs]
        outputs = model(inputs, output_ind=topvar, batch_size=test_bs)
        loss = criterion(*outputs, labels)
        correct = (dist(*outputs)>0).long() == labels
        num_correct = torch.nonzero(correct.data).size()
        if len(num_correct):
            total_correct += num_correct[0]
        correct = (-(dist(*outputs)<=0).float()).long() == labels
        num_correct = torch.nonzero(correct.data).size()
        if len(num_correct):
            total_correct += num_correct[0]
        total_loss += loss
        total_iters += 1

    return total_loss, total_correct / (total_iters*test_bs)

def get_embeddings(model, ds: CnfDataset, **kwargs):
    test_bs = 100
    all_labels = []
    all_encs = []
    # test_bs = settings['batch_size']
    settings = kwargs['settings'] if 'settings' in kwargs.keys() else CnfSettings()    
    sampler = torch.utils.data.sampler.SequentialSampler(ds)
    vloader = torch.utils.data.DataLoader(ds, batch_size=test_bs, sampler = sampler)
    total_iters = 0
    print('Begin forward embedding, number of mini-batches is %d' % len(vloader))

    for data in vloader:
        inputs = (data['variables'], data['clauses'])
        if  len(inputs[0]) != test_bs:
            print('Short batch!')            
        topvar = torch.abs(Variable(data['topvar'], requires_grad=False))
        labels = Variable(data['label'], requires_grad=False)
        if settings.hyperparameters['cuda']:
                topvar, labels = topvar.cuda(), labels.cuda()
                inputs = [t.cuda() for t in inputs]        
        outputs, aux_losses = model.encoder(inputs, output_ind=topvar, batch_size=test_bs)
        enc = model.embedder(outputs,output_ind=topvar, batch_size=len(inputs[0]))
        if settings['cuda']:
            all_encs.append(enc.data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
        else:
            all_encs.append(enc.data.numpy())
            all_labels.append(labels.data.numpy())
        total_iters += 1
        print('Done with iter %d' % total_iters)
        

    return (np.concatenate(all_encs,axis=0), np.concatenate(all_labels))
