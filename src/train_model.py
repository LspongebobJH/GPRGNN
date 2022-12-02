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

    # permute_masks = random_planetoid_splits
    # data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = appnp_net.to(device), data.to(device)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--train_rate', type=float, default=0.6)
    # parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str,
                        choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'],
                        default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop',
                        choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)

    parser.add_argument('--dataset', default='film')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--RPMAX', type=int, default=10)
    parser.add_argument('--net', type=str, choices=['GCN', 'GAT', 'APPNP', 'ChebNet', 'JKNet', 'GPRGNN'],
                        default='GPRGNN')

    args = parser.parse_args()

# def pipe(config:dict):
#     args = argparse.Namespace(**config)

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

    RPMAX = args.RPMAX
    Init = args.Init

    Gamma_0 = None
    alpha = args.alpha
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
