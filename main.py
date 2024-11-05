import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import PCQM4Mv2, ZINC
from ogb.graphproppred import PygGraphPropPredDataset
from torch.utils.data import Subset
from torch.utils.data import random_split
from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, eval_mae, load_fixed_splits, eval_rocauc_graph
from eval import *
from parse import parse_method, parser_add_main_args


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility (not necessary in this context)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node and Graph Classification/Regression')
parser_add_main_args(parser)
args = parser.parse_args()
if not args.global_dropout:
    args.global_dropout = args.dropout
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
print(f"DEVICE: {device}")
### Load and preprocess data ###
if args.dataset not in ('zinc', 'pcqm', 'ogbg-molhiv'):
    dataset = load_dataset(args.data_dir, args.dataset)

# if hasattr(dataset, 'label') and len(dataset.label.shape) == 1:
#     dataset.label = dataset.label.unsqueeze(1)
# if hasattr(dataset, 'label'):
#     dataset.label = dataset.label.to(device)
if args.dataset not in ('zinc', 'pcqm', 'ogbg-molhiv'):
    split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

    ### Check if the dataset is node-level or graph-level ###
    is_graph_level = hasattr(dataset, 'graphs')

if args.dataset not in ('zinc', 'pcqm', 'ogbg-molhiv'):
    if hasattr(dataset, 'label') and len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    if hasattr(dataset, 'label'):
        dataset.label = dataset.label.to(device)
    ### Node-level dataset ###
    ### Basic information of datasets ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    print(f"Dataset {args.dataset} | Num nodes {n} | Num edges {e} | Num node feats {d} | Num classes {c}")

    # Process edge attributes
    if 'edge_attr' in dataset.graph and dataset.graph['edge_attr'] is not None:
        edge_attr = dataset.graph['edge_attr']
        # Make the graph undirected while handling edge attributes
        dataset.graph['edge_index'], edge_attr = to_undirected(dataset.graph['edge_index'], edge_attr=edge_attr)
        # Remove self-loops
        dataset.graph['edge_index'], edge_attr = remove_self_loops(dataset.graph['edge_index'], edge_attr=edge_attr)
        # Add self-loops
        dataset.graph['edge_index'], edge_attr = add_self_loops(dataset.graph['edge_index'], edge_attr=edge_attr, num_nodes=n)
        dataset.graph['edge_attr'] = edge_attr
        edge_attr_dim = dataset.graph['edge_attr'].shape[1]
        dataset.graph['edge_attr'] = dataset.graph['edge_attr'].to(device)
    else:
        # If no edge attributes, proceed as before
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
        edge_attr_dim = None

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

    ### Load method ###
    model = parse_method(args, n, c, d, edge_attr_dim, device, task='node')

    ### Loss function (Single-class, Multi-class) ###
    if args.dataset in ('questions'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()

    ### Performance metric (Acc, AUC) ###
    if args.metric == 'rocauc':
        eval_func = eval_rocauc
    else:
        eval_func = eval_acc

    logger = Logger(args.runs, args)

    model.train()
    print('MODEL:', model)

    ### Training loop ###
    for run in range(args.runs):
        if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo'):
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        model._global = False
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        best_val = float('-inf')
        best_test = float('-inf')
        if args.save_model:
            save_model(args, model, optimizer, run)

        for epoch in range(args.local_epochs + args.global_epochs):
            if epoch == args.local_epochs:
                print("Start global attention!")
                if args.save_model:
                    model, optimizer = load_model(args, model, optimizer, run)
                model._global = True
            model.train()
            optimizer.zero_grad()

            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'], dataset.graph.get('edge_attr', None))
            if args.dataset in ('questions'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label
                loss = criterion(out[train_idx], true_label.squeeze(1)[
                    train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(
                    out[train_idx], dataset.label.squeeze(1)[train_idx])
            loss.backward()
            optimizer.step()

            result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                best_test = result[2]
                if args.save_model:
                    save_model(args, model, optimizer, run)

            if epoch % args.display_step == 0:
                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * result[0]:.2f}%, '
                      f'Valid: {100 * result[1]:.2f}%, '
                      f'Test: {100 * result[2]:.2f}%, '
                      f'Best Valid: {100 * best_val:.2f}%, '
                      f'Best Test: {100 * best_test:.2f}%')
        logger.print_statistics(run)

    results = logger.print_statistics()
    ### Save results ###
    save_result(args, results)

else:
    ### Graph-level dataset ###
    print("Graph Task")
    print(f"Dataset {args.dataset}")

    if args.dataset == "zinc":
        train_dataset = ZINC(root=f'data/ZINC', split="train", subset=True)
        valid_dataset = ZINC(root=f'data/ZINC', split="val", subset=True)
        test_dataset = ZINC(root=f'data/ZINC', split="test", subset=True)
        eval_func = eval_mae  # Mean Absolute Error
        criterion = nn.L1Loss()
        out_channels = 1  # For regression tasks like ZINC and PCQM4Mv2

    elif args.dataset == "pcqm":
        train_dataset = PCQM4Mv2(root=f'data/pcqm', split="train")
        valid_dataset = PCQM4Mv2(root=f'data/pcqm', split="val")
        test_dataset = PCQM4Mv2(root=f'data/pcqm', split="test")
        # Calcular el tamaño del subconjunto
        total_length = len(train_dataset)
        subset_length = int(total_length * 0.5)

        # Seleccionar aleatoriamente 1/3 de los índices
        indices = random.sample(range(total_length), subset_length)

        train_dataset = Subset(train_dataset, indices)
        eval_func = eval_mae  # Mean Absolute Error
        criterion = nn.L1Loss()
        out_channels = 1  # For regression tasks like ZINC and PCQM4Mv2

    elif args.dataset == "ogbg-molhiv":
        dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root='data/')
        split_idx = dataset.get_idx_split() 
        train_dataset = dataset[split_idx["train"]]
        valid_dataset = dataset[split_idx["valid"]]
        test_dataset = dataset[split_idx["test"]]
        eval_func = eval_rocauc_graph
        #out_channels = train_dataset.num_classes
        #criterion = nn.NLLLoss()
        criterion = nn.BCEWithLogitsLoss()
        out_channels = 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Train graphs: {len(train_dataset)}")
    print(f"Val graphs: {len(valid_dataset)}")
    print(f"Test graphs: {len(test_dataset)}\n")
    print(f"Batch size_ {args.batch_size}")
    print(f"Train dataloader: {len(train_loader)}")
    print(f"Val dataloader: {len(valid_loader)}")
    print(f"Test dataloader: {len(test_loader)}\n")

    # Determine the dimensionality of features and edge attributes
    in_channels = train_dataset.num_features
    try:
        edge_attr_dim = train_dataset.num_edge_features
    except:
        edge_attr_dim = None
        print("No edge attributes")

    ### Load method ###
    print(f"in channels: {in_channels}")
    print(f"out_channels: {out_channels}")
    print(f"edge dim: {edge_attr_dim}")
    print(f"use edges: {args.use_edges}")
    model = parse_method(args, None, out_channels, in_channels, edge_attr_dim, device, task='graph')
    total_params = count_parameters(model)
    print("Total de parámetros en el modelo:", total_params)
    
    model.train()
    print('MODEL:', model)
    print(f'Model device: {next(model.parameters()).device}')
    ### Training loop ###
    for run in range(args.runs):
        model.reset_parameters()
        model._global = False
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
        if args.dataset == "ogbg-molhiv":
            best_val = float('-inf')
            best_test = float('-inf')
        else: 
            best_val = float('inf')
            best_test = float('inf')
        if args.save_model:
            save_model(args, model, optimizer, run)

        for epoch in range(args.local_epochs + args.global_epochs):
            if epoch == args.local_epochs:
                print("Start global attention!")
                if args.save_model:
                    model, optimizer = load_model(args, model, optimizer, run)
                model._global = True
            model.train()
            total_loss = 0
            for i, batch in enumerate(train_loader):
                # if (args.dataset == "pcqm") and (i % 200 == 0):
                #     print(f"Batch {i}/{len(train_loader)}")
                if torch.isnan(batch.y).any():
                    print("NaNs encontrados en batch.y")
                    print(batch.y)
                    input("stop")

                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                target = batch.y.clone().detach()#.to(out.device)
                if args.dataset != "ogbg-molhiv":
                    loss = criterion(out.view(-1), target.view(-1))
                else:
                    # out = F.log_softmax(out, dim=1)
                    # target = target.view(-1).long()
                    target = batch.y.float().view(-1)#.to(device)
                    out = out.view(-1)
   
                    loss = criterion(out, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs

            train_score = evaluate_graph(model, train_loader, device, eval_func)
            valid_score = evaluate_graph(model, valid_loader, device, eval_func)
            test_score = evaluate_graph(model, test_loader, device, eval_func)

            if args.dataset != "ogbg-molhiv":
                if valid_score < best_val:
                    best_val = valid_score
                    best_test = test_score
                    if args.save_model:
                        save_model(args, model, optimizer, run)

                if (epoch % args.display_step == 0):
                    print(f'Epoch: {epoch:02d}, '
                        f'Loss: {total_loss / len(train_loader.dataset):.4f}, '
                        f'Train MAE: {train_score:.4f}, '
                        f'Valid MAE: {valid_score:.4f}, '
                        f'Test MAE: {test_score:.4f}, '
                        f'Best Valid MAE: {best_val:.4f}, '
                        f'Best Test MAE: {best_test:.4f}')
            else:
                if valid_score > best_test:
                    best_val = valid_score
                    best_test = test_score
                    if args.save_model:
                        save_model(args, model, optimizer, run)
                if (epoch % args.display_step == 0) and (args.dataset == "ogbg-molhiv"):
                    print(f'Epoch: {epoch:02d}, '
                        f'Loss: {total_loss / len(train_loader.dataset):.4f}, '
                        f'Train: {100 * train_score:.2f}%, '
                        f'Valid: {100 * valid_score:.2f}%, '
                        f'Test: {100 * test_score:.2f}%, '
                        f'Best Valid: {100 * best_val:.2f}%, '
                        f'Best Test: {100 * best_test:.2f}%')
        
        #model, optimizer = load_model(args, model, optimizer, run)
        train_best_score = evaluate_graph(model, train_loader, device, eval_func)
        valid_best_score = evaluate_graph(model, valid_loader, device, eval_func)
        test_best_score = evaluate_graph(model, test_loader, device, eval_func)
        print(f'Train: {100 * train_best_score:.2f}%, '
                f'Valid: {100 * valid_best_score:.2f}%, '
                f'Test: {100 * test_best_score:.2f}%')
        logger = Logger(args.runs, args)
        logger.add_result(run, [train_best_score, valid_best_score, test_best_score])
        #logger.print_statistics(run)
        results = [train_best_score, valid_best_score, test_best_score]

    # results = logger.print_statistics()
    ### Save results ###
    save_result(args, results)
