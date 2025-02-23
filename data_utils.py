import torch
import torch.nn.functional as F
from torch_geometric.datasets import HeterophilousGraphDataset, WikiCS
from ogb.graphproppred import Evaluator
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, mean_squared_error


def load_fixed_splits(data_dir, dataset, name):
    splits_lst = []
    if name in ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:
        torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(data.val_mask[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:,i])[0]
            splits_lst.append(splits)
    elif name in ['wikics']:
        torch_dataset = WikiCS(root=f"{data_dir}/wikics/")
        data = torch_dataset[0]
        for i in range(data.train_mask.shape[1]):
            splits = {}
            splits['train'] = torch.where(data.train_mask[:,i])[0]
            splits['valid'] = torch.where(torch.logical_or(data.val_mask, data.stopping_mask)[:,i])[0]
            splits['test'] = torch.where(data.test_mask[:])[0]
            splits_lst.append(splits)
    elif name in ['amazon-computer', 'amazon-photo', 'coauthor-cs', 'coauthor-physics']:
        splits = {}
        idx = np.load(f'{data_dir}/{name}_split.npz')
        splits['train'] = torch.from_numpy(idx['train'])
        splits['valid'] = torch.from_numpy(idx['valid'])
        splits['test'] = torch.from_numpy(idx['test'])
        splits_lst.append(splits)
    elif name in ['pokec']:
        split = np.load(f'{data_dir}/{name}/{name}-splits.npy', allow_pickle=True)
        for i in range(split.shape[0]):
            splits = {}
            splits['train'] = torch.from_numpy(np.asarray(split[i]['train']))
            splits['valid'] = torch.from_numpy(np.asarray(split[i]['valid']))
            splits['test'] = torch.from_numpy(np.asarray(split[i]['test']))
            splits_lst.append(splits)
    elif name in ['zinc', 'pcqm4mv2']:
        # For graph-level datasets, splits are provided by the dataset
        split_idx = dataset.get_idx_split()
        splits = {}
        splits['train'] = split_idx['train']
        splits['valid'] = split_idx['valid']
        splits['test'] = split_idx['test']
        splits_lst.append(splits)
    else:
        raise NotImplementedError

    return splits_lst

def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list)/len(acc_list)

def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct))/len(correct))

    return sum(acc_list)/len(acc_list)

## Regression
def eval_rmse(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        acc_list.append(rmse)

    return sum(acc_list)/len(acc_list)

def eval_mse(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    
    for i in range(y_true.shape[0]):
        mse = mean_squared_error(y_true, y_pred[:, i])
        acc_list.append(mse)

    return sum(acc_list)/len(acc_list)

def eval_mae(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    # print("TRUE:\n")
    # print(y_true)
    # print("PRED:\n")
    # print(y_pred)
    # print(y_true.shape)
    # print(y_pred.shape)
    #print(y_pred.unsqueeze(1).shape)
    #print(y_pred.unsqueeze(1))
    # print("\n")
    # print(y_true[:20])
    # print(y_pred[:20])
    return mean_absolute_error(y_true, y_pred)
    for i in range(y_true.shape[0]):
        mae = mean_absolute_error(y_true, y_pred)
        acc_list.append(mae)

    return sum(acc_list)/len(acc_list)


def eval_graph_dataset(dataset_name, y_true, y_pred):
    if dataset_name == 'zinc':
        return eval_mae(y_true, y_pred)
    elif dataset_name == 'pcqm4mv2':
        evaluator = Evaluator(name='pcqm4m-v2')
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        result_dict = evaluator.eval(input_dict)
        return result_dict['mae']
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")


def eval_rocauc_graph(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy().reshape(-1)
    # y_pred = (torch.sigmoid(y_pred) > 0.5).float()
    # y_pred = y_pred.detach().cpu().numpy().reshape(-1)
    y_pred = torch.sigmoid(y_pred).detach().cpu().numpy().reshape(-1)

    # if np.isnan(y_true).any():
    #     print("NaNs encontrados en y_true")
    #     print(y_true)
    # if np.isnan(y_pred).any():
    #     print("NaNs encontrados en y_pred")
    #     print(y_pred)
    return roc_auc_score(y_true, y_pred)

def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:,1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)

dataset_drive_url = {
    'snap-patents' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}

