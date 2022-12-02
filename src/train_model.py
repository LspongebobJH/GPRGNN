# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# Distributed under terms of the MIT license.

"""

"""

import argparse
from dataset_utils import DataLoader
from utils import random_planetoid_splits
from GNN_models import *

import torch
import torch.nn.functional as F
from tqdm import tqdm

import numpy as np
from ray import tune
import wandb

from init_layers import init_layers


def RunExp(args, dataset, data, Net):

    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    appnp_net = Net(dataset, args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = appnp_net.to(device), data.to(device)

    # permute_masks = random_planetoid_splits
    # data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    if args.init in ['nimfor', 'nimback']:
        init_layers(data, appnp_net, args.init)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)

    test_acc = 0
    cur_step = 0
    vlss_mn = np.inf
    vacc_mx = 0.0
    for epoch in range(args.epochs):
        train(model, optimizer, data)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        wandb.log({'train_acc': train_acc, 'valid_acc': val_acc, 'test_acc': tmp_test_acc})
        wandb.log({'train_loss': train_loss, 'valid_loss': val_loss, 'test_loss': tmp_test_loss})
    
        
        if val_acc >= vacc_mx or val_loss <= vlss_mn:
            if val_acc >= vacc_mx and val_loss <= vlss_mn:
                test_acc = tmp_test_acc
            vacc_mx = np.max((val_acc, vacc_mx))
            vlss_mn = np.min((val_loss, vlss_mn))
            cur_step = 0
        else:
            cur_step += 1
            if cur_step >= args.early_stopping:
                break

        print(
            "Epoch {:05d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Test Acc {:4f} | Patience {}/{}".format(
                epoch, train_loss.item(), train_acc, val_loss, val_acc, tmp_test_acc, cur_step, args.early_stopping))  

    return test_acc, vacc_mx

def pipe(config:dict):
    exp = config['exp']
    name = config['dataset']+'/'+config['init']
    wandb.init(project=f"exp{exp}", config=config, dir='/mnt/jiahanli/wandb', name=name)

    args = argparse.Namespace(**config)

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'ChebNet':
        Net = ChebNet
    elif gnn_name == 'JKNet':
        Net = GCN_JKNet
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN

    dname = args.dataset
    dataset, data = DataLoader(dname, args.split)

    # RPMAX = args.RPMAX
    # Init = args.Init

    Gamma_0 = None
    # alpha = args.alpha
    # train_rate = args.train_rate
    # val_rate = args.val_rate
    # percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    # val_lb = int(round(val_rate*len(data.y)))
    # TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)
    # print('True Label rate: ', TrueLBrate)

    args.C = len(data.y.unique())
    args.Gamma = Gamma_0

    # for RP in tqdm(range(RPMAX)):

        # test_acc, best_val_acc, Gamma_0 = RunExp(
        #     args, dataset, data, Net, percls_trn, val_lb)
    test_acc, best_val_acc = RunExp(
        args, dataset, data, Net)

    print(f'{gnn_name} on dataset {args.dataset}, the dataset split is {args.split}:')
    print(f'test acc = {test_acc:.4f} | best val acc = {best_val_acc:.4f}')

    wandb.log({'final_test_acc': test_acc})  
    wandb.finish()

    return test_acc

def tune_pipe(config):
    test_acc = pipe(config)
    tune.report(test_acc=test_acc)

def run_ray():
    exp = 69
    num_samples = 1
    searchSpace = {
        'dataset': tune.grid_search(['texas', 'wisconsin', 'film', 'squirrel', 'chameleon', 'cornell']),
        'split': tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        'epochs':5000,
        'lr':tune.grid_search([1e-2, 5e-2, 1e-3, 5e-3]),
        'weight_decay':tune.grid_search([5e-4, 0.0]),
        'early_stopping':200,
        'hidden':tune.grid_search([64, 128]),
        'dropout':tune.grid_search([0.0, 0.5, 0.7]),
        'K':10,
        'alpha':tune.grid_search([0.1, 0.2, 0.5, 0.9]),
        'C':0,
        'Init':'PPR',
        'Gamma':None,
        'ppnp':'GPR_prop',
        'heads':8,
        'output_heads':1,            
        'net':'GPRGNN',
        'init': 'nimfor',
        'exp': exp
    }
    
    print(searchSpace)

    analysis=tune.run(tune_pipe, config=searchSpace, name=f"{exp}", num_samples=num_samples, \
        resources_per_trial={'cpu': 12, 'gpu':1}, log_to_file=f"out.log", \
        local_dir="/mnt/jiahanli/nim_output", max_failures=1)

def run_test():
    searchSpace = {
        'dataset':'cornell',
        'split':0,
        'epochs':5000,
        'lr':0.002,
        'weight_decay':0.0005,
        'early_stopping':200,
        'hidden':64,
        'dropout':0.5,
        'K':10,
        'alpha':0.1,
        'C':0,
        'Init':'PPR',
        'Gamma':None,
        'ppnp':'GPR_prop',
        'heads':8,
        'output_heads':1,            
        'net':'GPRGNN',
        'init': 'nimfor',
        'exp': 500
    }

    print(searchSpace)
    pipe(searchSpace)

if __name__ == "__main__":
    run='test'
    
    if run == 'test':
        run_test()
    elif run == 'ray':
        run_ray()
    
    print(1)