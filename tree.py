"""
Main script for training an adaptive neural tree (ANT).
"""
from __future__ import print_function
import argparse
import os
import sys
import json
import time
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

import matplotlib
matplotlib.use('agg')

from data import get_dataloaders, get_dataset_details, normalize_dataset_name
from models import Tree, One
from ops import get_params_node
from utils import define_node, get_scheduler, set_random_seed
from visualisation import visualise_routers_behaviours
from training_config import get_training_config


# Experiment settings (CLI is intentionally minimal)
parser = argparse.ArgumentParser(description='Adaptive Neural Trees')
parser.add_argument('--experiment', '-e', dest='experiment', default='tree', help='experiment name')
parser.add_argument('--dataset', default='mnist', help='dataset code (e.g. mnist, cifar10, letter)')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')

args = parser.parse_args()
args.dataset = normalize_dataset_name(args.dataset)

# Load all training/model settings from dataset-specific config.
cfg = get_training_config(args.dataset)

# Non-config runtime defaults.
args.subexperiment = ''
args.no_cuda = not cfg.get('use_gpu', True)
args.gpu = ''
# args.seed is set via CLI argument (--seed) above with default value 0
args.num_workers = 0
args.log_interval = 10
args.augmentation_on = False
args.momentum = 0.5
args.criteria = 'avg_valid_loss'
args.epochs_finetune_node = 1

# Dataset-config driven parameters.
args.lr = cfg['learning_rate']
args.batch_size = cfg['batch_size']
args.epochs_node = cfg['epochs_node']
args.epochs_finetune = cfg['epochs_finetune']
args.epochs_patience = cfg['epochs_patience']
args.maxdepth = cfg['maxdepth']
args.scheduler = cfg['scheduler']
args.valid_ratio = cfg['valid_ratio']

args.router_ver = cfg['router_ver']
args.router_ngf = cfg['router_ngf']
args.router_k = cfg['router_k']
args.router_dropout_prob = cfg['router_dropout_prob']

args.transformer_ver = cfg['transformer_ver']
args.transformer_ngf = cfg['transformer_ngf']
args.transformer_k = cfg['transformer_k']
args.transformer_expansion_rate = cfg['transformer_expansion_rate']
args.transformer_reduction_rate = cfg['transformer_reduction_rate']

args.solver_ver = cfg['solver_ver']
args.solver_inherit = cfg['solver_inherit']
args.solver_dropout_prob = cfg['solver_dropout_prob']

args.downsample_interval = cfg['downsample_interval']
args.batch_norm = cfg['batch_norm']
args.finetune_during_growth = cfg['finetune_during_growth']
args.visualise_split = cfg['visualization_split']
args.ensemble = cfg.get('ensemble', not cfg.get('no_ensemble', False))

# GPUs devices:
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Set the seed for repeatability
set_random_seed(args.seed, args.cuda)

# Define a dictionary for post-inspection of the model:
records = vars(args)
records['time'] = 0.0
records['counter'] = 0  # number of optimization steps

records['train_nodes'] = []  # node indices for each logging interval
records['train_loss'] = []   # avg. train. loss for each log interval
records['train_best_loss'] = np.inf  # best train. loss
records['train_epoch_loss'] = []  # epoch wise train loss

records['valid_nodes'] = []
records['valid_best_loss_nodes'] = []
records['valid_best_loss_nodes_split'] = []
records['valid_best_loss_nodes_ext'] = []
records['valid_best_root_nosplit'] = np.inf
records['valid_best_loss'] = np.inf
records['valid_best_accuracy'] = 0.0
records['valid_epoch_loss'] = []
records['valid_epoch_accuracy'] = []

records['test_best_loss'] = np.inf
records['test_best_accuracy'] = 0.0
records['test_epoch_loss'] = []
records['test_epoch_accuracy'] = []

# Final soft and hard inference results for reporting
records['final_soft_train_accuracy'] = 0.0
records['final_hard_train_accuracy'] = 0.0
records['final_soft_valid_accuracy'] = 0.0
records['final_hard_valid_accuracy'] = 0.0
records['final_soft_test_accuracy'] = 0.0
records['final_hard_test_accuracy'] = 0.0
records['final_trainable_params'] = 0


# -----------------------------  Data loaders ---------------------------------
train_loader, valid_loader, test_loader, NUM_TRAIN, NUM_VALID = get_dataloaders(
    args.dataset, args.batch_size, args.augmentation_on,
    cuda=args.cuda, num_workers=args.num_workers,
)
args.input_nc, args.input_width, args.input_height, args.classes = \
    get_dataset_details(args.dataset)
args.no_classes = len(args.classes)


# -----------------------------  Components ----------------------------------
def print_model_parameter_stats(model, title='Model'):
    """Print total and trainable parameter counts for a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("{} parameters: total={}, trainable={}".format(
        title,
        "{:,}".format(total_params),
        "{:,}".format(trainable_params),
    ))


def train(model, data_loader, optimizer, node_idx):
    """ Train step"""
    model.train()
    train_loss = 0
    no_points = 0
    train_epoch_loss = 0
    optimizer_switched = False

    # train the model
    for batch_idx, (x, y) in enumerate(data_loader):
        optimizer.zero_grad()
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)
        y_pred, p_out = model(x)

        loss = F.nll_loss(y_pred, y)
        train_epoch_loss += loss.item() * y.size(0)
        train_loss += loss.item() * y.size(0)
        loss.backward()

        # Adam allocates optimizer states lazily on step(); if this OOMs on GPU,
        # fallback to SGD (lower optimizer-state memory footprint) and continue.
        try:
            optimizer.step()
        except RuntimeError as e:
            err = str(e).lower()
            is_cuda_oom = 'out of memory' in err and 'cuda' in err
            if is_cuda_oom and isinstance(optimizer, optim.Adam):
                print("\nCUDA OOM during Adam step; switching optimizer to SGD for lower memory usage.")
                if args.cuda:
                    torch.cuda.empty_cache()
                params = [p for group in optimizer.param_groups for p in group['params']]
                optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
                optimizer.step()
                optimizer_switched = True
            else:
                raise

        records['counter'] += 1
        no_points += y.size(0)

        if batch_idx % args.log_interval == 0:
            # show the interval-wise average loss:
            train_loss /= no_points
            records['train_loss'].append(train_loss)
            records['train_nodes'].append(node_idx)

            sys.stdout.flush()
            sys.stdout.write('\t      [{}/{} ({:.0f}%)]      Loss: {:.6f} \r'.
                    format(batch_idx*len(x), NUM_TRAIN,
                    100. * batch_idx / NUM_TRAIN, train_loss))

            train_loss = 0
            no_points = 0

    # compute average train loss for the epoch
    train_epoch_loss /= NUM_TRAIN
    records['train_epoch_loss'].append(train_epoch_loss)
    if train_epoch_loss < records['train_best_loss']:
        records['train_best_loss'] = train_epoch_loss

    print('\nTrain set: Average loss: {:.4f}'.format(train_epoch_loss))
    return optimizer, optimizer_switched


def valid(model, data_loader, node_idx, struct):
    """ Validation step """
    model.eval()
    valid_epoch_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            # sum up batch loss
            valid_epoch_loss += F.nll_loss(
                output, target, size_average=False,
            ).item()

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    valid_epoch_loss /= NUM_VALID
    valid_epoch_accuracy = 100. * correct / NUM_VALID
    records['valid_epoch_loss'].append(valid_epoch_loss)
    records['valid_epoch_accuracy'].append(valid_epoch_accuracy)

    if valid_epoch_loss < records['valid_best_loss']:
        records['valid_best_loss'] = valid_epoch_loss

    if valid_epoch_accuracy > records['valid_best_accuracy']:
        records['valid_best_accuracy'] = valid_epoch_accuracy

    # see if the current node is root and undergoing the initial training
    # prior to the growth phase.
    is_init_root_train = not model.split and not model.extend and node_idx == 0

    # save the best split model during node-wise training as model_tmp.pth
    if not is_init_root_train and model.split and \
            valid_epoch_loss < records['valid_best_loss_nodes_split'][node_idx]:
        records['valid_best_loss_nodes_split'][node_idx] = valid_epoch_loss
        checkpoint_model('model_tmp.pth', model=model)
        checkpoint_msc(struct, records)

    # save the best extended model during node-wise training as model_ext.pth
    if not is_init_root_train and model.extend and \
            valid_epoch_loss < records['valid_best_loss_nodes_ext'][node_idx]:
        records['valid_best_loss_nodes_ext'][node_idx] = valid_epoch_loss
        checkpoint_model('model_ext.pth', model=model)
        checkpoint_msc(struct, records)

    # separately store best performance for the initial root training
    if is_init_root_train \
            and valid_epoch_loss < records['valid_best_root_nosplit']:
        records['valid_best_root_nosplit'] = valid_epoch_loss
        checkpoint_model('model_tmp.pth', model=model)
        checkpoint_msc(struct, records)

    # saving model during the refinement (fine-tuning) phase
    if not is_init_root_train and \
            valid_epoch_loss < records['valid_best_loss_nodes'][node_idx]:
        records['valid_best_loss_nodes'][node_idx] = valid_epoch_loss
        if not model.split and not model.extend:
            checkpoint_model('model_tmp.pth', model=model)
            checkpoint_msc(struct, records)

    end = time.time()
    records['time'] = end - start
    print(
        'Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
        '\nTook {} seconds. '.format(
            valid_epoch_loss, correct, NUM_VALID,
            100. * correct / NUM_VALID, records['time'],
        ),
    )
    return valid_epoch_loss


def test(model, data_loader):
    """ Test step """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(data_loader.dataset)
    test_accuracy = 100. * correct / len(data_loader.dataset)
    records['test_epoch_loss'].append(test_loss)
    records['test_epoch_accuracy'].append(test_accuracy)

    if test_loss < records['test_best_loss']:
        records['test_best_loss'] = test_loss

    if test_accuracy > records['test_best_accuracy']:
        records['test_best_accuracy'] = test_accuracy

    end = time.time()
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
        '\nTook {} seconds. '.format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset), end - start,
        ),
    )


def evaluate_model_accuracy(model, data_loader, mode_name='soft', use_cuda=False):
    """Evaluate model accuracy on a dataset in soft or hard inference mode.
    
    Args:
        model: Tree model to evaluate
        data_loader: DataLoader with (x, y) batches
        mode_name: Either 'soft' (multi-path) or 'hard' (single-path greedy)
    
    Returns:
        accuracy: Percentage accuracy on the dataset
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total if total > 0 else 0.0
    return accuracy


def evaluate_and_record_final_results(model, tree_struct, train_loader, valid_loader, test_loader, start_time=None):
    """Evaluate model in both soft and hard modes and record final accuracies.
    
    Args:
        model: Tree model (in soft mode)
        tree_struct: Tree structure info
        train_loader, valid_loader, test_loader: DataLoaders
        start_time: Start time of training (for elapsed time calculation)
    """
    from utils import load_tree_model
    
    # Calculate total training time and final tree depth.
    end_time = time.time()
    total_train_time = end_time - start_time if start_time is not None else 0
    train_hours = int(total_train_time // 3600)
    train_mins = int((total_train_time % 3600) // 60)
    train_secs = int(total_train_time % 60)
    final_tree_depth = max([node['level'] for node in tree_struct]) if tree_struct else 0
    final_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    records['final_tree_depth'] = final_tree_depth
    records['final_trainable_params'] = final_trainable_params
    
    print("\n" + "="*60)
    print("Final Inference Evaluation: Soft vs Hard")
    print("="*60)
    print(f"Total Training Time: {train_hours:02d}h {train_mins:02d}m {train_secs:02d}s ({total_train_time:.1f}s)")
    print(f"Final Tree Depth: {final_tree_depth}")
    print(f"Final Trainable Parameters: {final_trainable_params:,}")
    
    model_path = "./experiments/{}/{}/{}/checkpoints/model.pth".format(
        args.dataset, args.experiment, args.subexperiment
    )
    perf_path = "./experiments/{}/{}/{}/checkpoints/performance.txt".format(
        args.dataset, args.experiment, args.subexperiment
    )
    
    # Keep GPU memory for training only: run final evaluation on CPU.
    if args.cuda:
        model = model.cpu()
        torch.cuda.empty_cache()

    print("Final evaluation device: CPU (GPU reserved for training only)")

    # Evaluate in SOFT mode (multi-path, default)
    print("\n[SOFT INFERENCE - Multi-path]")
    soft_train_acc = evaluate_model_accuracy(model, train_loader, 'soft', use_cuda=False)
    soft_valid_acc = evaluate_model_accuracy(model, valid_loader, 'soft', use_cuda=False)
    soft_test_acc = evaluate_model_accuracy(model, test_loader, 'soft', use_cuda=False)
    
    print(f"  Train accuracy: {soft_train_acc:.2f}%")
    print(f"  Valid accuracy: {soft_valid_acc:.2f}%")
    print(f"  Test  accuracy: {soft_test_acc:.2f}%")
    
    records['final_soft_train_accuracy'] = soft_train_acc
    records['final_soft_valid_accuracy'] = soft_valid_acc
    records['final_soft_test_accuracy'] = soft_test_acc
    
    hard_train_acc = None
    hard_valid_acc = None
    hard_test_acc = None
    
    # Evaluate in HARD mode (single-path greedy)
    print("\n[HARD INFERENCE - Single-path greedy]")
    try:
        model_hard = load_tree_model(
            model_path, cuda_on=False,
            soft_decision=False, stochastic=False, breadth_first=False
        )
        hard_train_acc = evaluate_model_accuracy(model_hard, train_loader, 'hard', use_cuda=False)
        hard_valid_acc = evaluate_model_accuracy(model_hard, valid_loader, 'hard', use_cuda=False)
        hard_test_acc = evaluate_model_accuracy(model_hard, test_loader, 'hard', use_cuda=False)
        
        print(f"  Train accuracy: {hard_train_acc:.2f}%")
        print(f"  Valid accuracy: {hard_valid_acc:.2f}%")
        print(f"  Test  accuracy: {hard_test_acc:.2f}%")
        
        records['final_hard_train_accuracy'] = hard_train_acc
        records['final_hard_valid_accuracy'] = hard_valid_acc
        records['final_hard_test_accuracy'] = hard_test_acc
    except Exception as e:
        print(f"  Error loading hard model: {e}")
        print("  Skipping hard inference evaluation.")
    
    print("\n" + "="*60)
    
    # Write results to performance.txt
    with open(perf_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Final Inference Performance Report\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Experiment: {args.experiment}\n")
        f.write(f"Final Tree Depth: {final_tree_depth}\n")
        f.write(f"Final Trainable Parameters: {final_trainable_params:,}\n")
        f.write(f"Training Time: {train_hours:02d}h {train_mins:02d}m {train_secs:02d}s\n")
        f.write(f"Training Time (seconds): {total_train_time:.1f}\n\n")
        
        f.write("[SOFT INFERENCE - Multi-path]\n")
        f.write(f"  Train accuracy: {soft_train_acc:.2f}%\n")
        f.write(f"  Valid accuracy: {soft_valid_acc:.2f}%\n")
        f.write(f"  Test  accuracy: {soft_test_acc:.2f}%\n\n")
        
        f.write("[HARD INFERENCE - Single-path greedy]\n")
        if hard_train_acc is not None:
            f.write(f"  Train accuracy: {hard_train_acc:.2f}%\n")
            f.write(f"  Valid accuracy: {hard_valid_acc:.2f}%\n")
            f.write(f"  Test  accuracy: {hard_test_acc:.2f}%\n")
        else:
            f.write("  Error loading hard model - skipped\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"\nPerformance results saved to {perf_path}")


def _load_checkpoint(model_file_name):
    save_dir = "./experiments/{}/{}/{}/{}".format(
        args.dataset, args.experiment, args.subexperiment, 'checkpoints',
    )
    model = torch.load(save_dir + '/' + model_file_name)
    if args.cuda:
        model.cuda()
    return model


def checkpoint_model(model_file_name, struct=None, modules=None, model=None, figname='hist.png', data_loader=None):
    if not(os.path.exists(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment))):
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'figures'))
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'checkpoints'))
    
    # If model is not given, then build one. 
    if not(model) and modules and struct:
        model = Tree(struct, modules, cuda_on=args.cuda)
        
    # save the model:
    save_dir = "./experiments/{}/{}/{}/{}".format(args.dataset, args.experiment, args.subexperiment, 'checkpoints')
    model_path = save_dir + '/' + model_file_name
    torch.save(model, model_path)
    print("Model saved to {}".format(model_path))

    # save tree histograms:
    if args.visualise_split and not(data_loader is None):
        save_hist_dir = "./experiments/{}/{}/{}/{}".format(args.dataset, args.experiment, args.subexperiment, 'figures')
        visualise_routers_behaviours(model, data_loader, fig_scale=6, axis_font=20, subtitle_font=20, 
                                     cuda_on=args.cuda, objects=args.classes, plot_on=False, 
                                     save_as=save_hist_dir + '/' + figname)


def checkpoint_msc(struct, data_dict):
    """ Save structural information of the model and experimental results.

    Args:
        struct (list) : list of dictionaries each of which contains
            meta information about each node of the tree.
        data_dict (dict) : data about the experiment (e.g. loss, configurations)
    """
    if not(os.path.exists(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment))):
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'figures'))
        os.makedirs(os.path.join("./experiments", args.dataset, args.experiment, args.subexperiment, 'checkpoints'))

    # save the tree structures as a json file:
    save_dir = "./experiments/{}/{}/{}/{}".format(args.dataset,args.experiment,args.subexperiment,'checkpoints')
    struct_path = save_dir + "/tree_structures.json"
    with open(struct_path, 'w') as f:
        json.dump(struct, f)
    print("Tree structure saved to {}".format(struct_path))

    # save the dictionary as jason file:
    dict_path = save_dir + "/records.json"
    with open(dict_path, 'w') as f_d:
        json.dump(data_dict, f_d)
    print("Other data saved to {}".format(dict_path))


def get_decision(criteria, node_idx, tree_struct):
    """ Define the splitting criteria

    Args:
        criteria (str): Growth criteria.
        node_idx (int): Index of the current node.
        tree_struct (list) : list of dictionaries each of which contains
            meta information about each node of the tree.

    Returns:
        The function returns one of the following strings
            'split': split the node
            'extend': extend the node
            'keep': keep the node as it is
    """
    if criteria == 'always':  # always split or extend
        if tree_struct[node_idx]['valid_accuracy_gain_ext'] > tree_struct[node_idx]['valid_accuracy_gain_split'] > 0.0:
            return 'extend'
        else:
            return 'split'
    elif criteria == 'avg_valid_loss':
        if tree_struct[node_idx]['valid_accuracy_gain_ext'] > tree_struct[node_idx]['valid_accuracy_gain_split'] and \
                        tree_struct[node_idx]['valid_accuracy_gain_ext'] > 0.0:
            print("Average valid loss is reduced by {} ".format(tree_struct[node_idx]['valid_accuracy_gain_ext']))
            return 'extend'

        elif tree_struct[node_idx]['valid_accuracy_gain_split'] > 0.0:
            print("Average valid loss is reduced by {} ".format(tree_struct[node_idx]['valid_accuracy_gain_split']))
            return 'split'

        else:
            print("Average valid loss is aggravated by split/extension."
                  " Keep the node as it is.")
            return 'keep'
    else:
        raise NotImplementedError(
            "specified growth criteria is not available. ",
        )


def optimize_fixed_tree(
        model, tree_struct, train_loader,
        valid_loader, test_loader, no_epochs, node_idx,
):
    """ Train a tree with fixed architecture.

    Args:
        model (torch.nn.module): tree model
        tree_struct (list): list of dictionaries which contain information
                            about all nodes in the tree.
        train_loader (torch.utils.data.DataLoader) : data loader of train data
        valid_loader (torch.utils.data.DataLoader) : data loader of valid data
        test_loader (torch.utils.data.DataLoader) : data loader of test data
        no_epochs (int): number of epochs for training
        node_idx (int): index of the node you want to optimize

    Returns:
        returns the trained model and newly added nodes (if grown).
    """
    # get if the model is growing or fixed
    grow = (model.split or model.extend)

    # define optimizer and trainable parameters
    params, names = get_params_node(grow, node_idx,  model)
    for i, (n, p) in enumerate(model.named_parameters()):
        if not(n in names):
            # print('(Fix)   ' + n)
            p.requires_grad = False
        else:
            # print('(Optimize)     ' + n)
            p.requires_grad = True

    for i, p in enumerate(params):
        if not(p.requires_grad):
            print("(Grad not required)" + names[i])

    print_model_parameter_stats(model, title='Current model')

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, params), lr=args.lr,
    )
    if args.scheduler:
        scheduler = get_scheduler(args.scheduler, optimizer, grow)

    # monitor nodewise best valid loss:
    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes']) == node_idx:
        records['valid_best_loss_nodes'].append(np.inf)
    
    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes_split']) == node_idx:
        records['valid_best_loss_nodes_split'].append(np.inf)

    if not(not(grow) and node_idx==0) and len(records['valid_best_loss_nodes_ext']) == node_idx:
        records['valid_best_loss_nodes_ext'].append(np.inf)

    # start training
    min_improvement = 0.0  # acceptable improvement in loss for early stopping
    valid_loss = np.inf
    patience_cnt = 1

    for epoch in range(1, no_epochs + 1):
        print("\n----- Layer {}, Node {}, Epoch {}/{}, Patience {}/{}---------".
              format(tree_struct[node_idx]['level'], node_idx, 
                     epoch, no_epochs, patience_cnt, args.epochs_patience))
        optimizer, optimizer_switched = train(model, train_loader, optimizer, node_idx)
        if optimizer_switched and args.scheduler:
            scheduler = get_scheduler(args.scheduler, optimizer, grow)
        valid_loss_new = valid(model, valid_loader, node_idx, tree_struct)
        
        # learning rate scheduling:
        if args.scheduler == 'plateau':
            scheduler.step(valid_loss_new)
        elif args.scheduler == 'step_lr':
            scheduler.step()
        
        test(model, test_loader)

        if not((valid_loss-valid_loss_new) > min_improvement) and grow:
            patience_cnt += 1
        valid_loss = valid_loss_new*1.0
        
        if patience_cnt > args.epochs_patience > 0:
            print('Early stopping')
            break
 
    # load the node-wise best model based on validation accuracy:
    if no_epochs > 0 and grow:
        if model.extend:
            print('return the node-wise best extended model')
            model = _load_checkpoint('model_ext.pth')
        else:
            print('return the node-wise best split model')
            model = _load_checkpoint('model_tmp.pth')

    # return the updated models:
    tree_modules = model.update_tree_modules()
    if model.split:
        child_left, child_right = model.update_children()
        return model, tree_modules, child_left, child_right
    elif model.extend:
        child_extension = model.update_children()
        return model, tree_modules, child_extension
    else:
        return model, tree_modules


def grow_ant_nodewise():
    """The main function for optimising an ANT """

    # ############## 0: Define the root node and optimise ###################
    # define the root node:
    tree_struct = []  # stores graph information for each node
    tree_modules = []  # stores modules for each node
    root_meta, root_module = define_node(
        args, node_index=0, level=0, parent_index=-1, tree_struct=tree_struct,
    )
    tree_struct.append(root_meta)
    tree_modules.append(root_module)

    # train classifier on root node (no split no extension):
    model = Tree(
        tree_struct, tree_modules, split=False, extend=False, cuda_on=args.cuda,
    )
    if args.cuda:
        model.cuda()

    # optimise
    model, tree_modules = optimize_fixed_tree(
        model, tree_struct,
        train_loader, valid_loader, test_loader, args.epochs_node, node_idx=0,
    )
    checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules)
    checkpoint_msc(tree_struct, records)

    if not args.ensemble:
        print("\nEnsemble mode disabled: skipping ANT growth (split/extend).")
        model_final = Tree(tree_struct, tree_modules, split=False, cuda_on=args.cuda)
        if args.cuda:
            model_final.cuda()
        print_model_parameter_stats(model_final, title='Final tree model')
        evaluate_and_record_final_results(
            model_final, tree_struct,
            train_loader, valid_loader, test_loader,
            start_time=start,
        )
        checkpoint_msc(tree_struct, records)
        return

    # ######################## 1: Growth phase starts ########################
    nextind = 1
    last_node = 0
    for lyr in range(args.maxdepth):
        print("---------------------------------------------------------------")
        print("\nAt layer " + str(lyr))
        for node_idx in range(len(tree_struct)):
            change = False
            if tree_struct[node_idx]['is_leaf'] and not(tree_struct[node_idx]['visited']):

                print("\nProcessing node " + str(node_idx))

                # -------------- Define children candidate nodes --------------
                # ---------------------- (1) Split ----------------------------
                # left child
                identity = True
                meta_l, node_l = define_node(
                    args,
                    node_index=nextind, level=lyr+1,
                    parent_index=node_idx, tree_struct=tree_struct,
                    identity=identity,
                )
                # right child
                meta_r, node_r = define_node(
                    args,
                    node_index=nextind+1, level=lyr+1,
                    parent_index=node_idx, tree_struct=tree_struct,
                    identity=identity,
                )
                # inheriting solver modules to facilitate optimization:
                if args.solver_inherit and meta_l['identity'] and meta_r['identity'] and not(node_idx == 0):
                    node_l['classifier'] = tree_modules[node_idx]['classifier']
                    node_r['classifier'] = tree_modules[node_idx]['classifier']

                # define a tree with a new split by adding two children nodes:
                model_split = Tree(tree_struct, tree_modules,
                                   split=True, node_split=node_idx,
                                   child_left=node_l, child_right=node_r,
                                   extend=False,
                                   cuda_on=args.cuda)

                # -------------------- (2) Extend ----------------------------
                # define a tree with node extension
                meta_e, node_e = define_node(
                    args,
                    node_index=nextind,
                    level=lyr+1,
                    parent_index=node_idx,
                    tree_struct=tree_struct,
                    identity=False,
                )
                # Set the router at the current node as one-sided One().
                # TODO: this is not ideal as it changes tree_modules
                tree_modules[node_idx]['router'] = One()

                # define a tree with an extended edge by adding a node
                model_ext = Tree(tree_struct, tree_modules,
                                 split=False,
                                 extend=True, node_extend=node_idx,
                                 child_extension=node_e,
                                 cuda_on=args.cuda)

                # ---------------------- Optimise -----------------------------
                best_tr_loss = records['train_best_loss']
                best_va_loss = records['valid_best_loss']
                best_te_loss = records['test_best_loss']

                print("\n---------- Optimizing a binary split ------------")
                if args.cuda:
                    model_split.cuda()

                # split and optimise
                model_split, tree_modules_split, node_l, node_r \
                    = optimize_fixed_tree(model_split, tree_struct,
                                          train_loader, valid_loader, test_loader,
                                          args.epochs_node,
                                          node_idx)

                best_tr_loss_after_split = records['train_best_loss']
                best_va_loss_adter_split = records['valid_best_loss_nodes_split'][node_idx]
                best_te_loss_after_split = records['test_best_loss']
                tree_struct[node_idx]['train_accuracy_gain_split'] \
                    = best_tr_loss - best_tr_loss_after_split
                tree_struct[node_idx]['valid_accuracy_gain_split'] \
                    = best_va_loss - best_va_loss_adter_split
                tree_struct[node_idx]['test_accuracy_gain_split'] \
                    = best_te_loss - best_te_loss_after_split

                print("\n----------- Optimizing an extension --------------")
                if not(meta_e['identity']):
                    if args.cuda:
                        model_ext.cuda()

                    # make deeper and optimise
                    model_ext, tree_modules_ext, node_e \
                        = optimize_fixed_tree(model_ext, tree_struct,
                                              train_loader, valid_loader, test_loader,
                                              args.epochs_node,
                                              node_idx)

                    best_tr_loss_after_ext = records['train_best_loss']
                    best_va_loss_adter_ext = records['valid_best_loss_nodes_ext'][node_idx]
                    best_te_loss_after_ext = records['test_best_loss']

                    # TODO: record the gain from split/extra depth:
                    #  need separately record best losses for split & depth
                    tree_struct[node_idx]['train_accuracy_gain_ext'] \
                        = best_tr_loss - best_tr_loss_after_ext
                    tree_struct[node_idx]['valid_accuracy_gain_ext'] \
                        = best_va_loss - best_va_loss_adter_ext
                    tree_struct[node_idx]['test_accuracy_gain_ext'] \
                        = best_te_loss - best_te_loss_after_ext
                else:
                    print('No extension as '
                          'the transformer is an identity function.')
                
                # ---------- Decide whether to split, extend or keep -----------
                criteria = get_decision(args.criteria, node_idx, tree_struct)

                if criteria == 'split':
                    print("\nSplitting node " + str(node_idx))
                    # update the parent node
                    tree_struct[node_idx]['is_leaf'] = False
                    tree_struct[node_idx]['left_child'] = nextind
                    tree_struct[node_idx]['right_child'] = nextind+1
                    tree_struct[node_idx]['split'] = True

                    # add the children nodes
                    tree_struct.append(meta_l)
                    tree_modules_split.append(node_l)
                    tree_struct.append(meta_r)
                    tree_modules_split.append(node_r)

                    # update tree_modules:
                    tree_modules = tree_modules_split
                    nextind += 2
                    change = True
                elif criteria == 'extend':
                    print("\nExtending node " + str(node_idx))
                    # update the parent node
                    tree_struct[node_idx]['is_leaf'] = False
                    tree_struct[node_idx]['left_child'] = nextind
                    tree_struct[node_idx]['extended'] = True

                    # add the children nodes
                    tree_struct.append(meta_e)
                    tree_modules_ext.append(node_e)

                    # update tree_modules:
                    tree_modules = tree_modules_ext
                    nextind += 1
                    change = True
                else:
                    # revert weights back to state before split
                    print("No splitting at node " + str(node_idx))
                    print("Revert the weights to the pre-split state.")
                    model = _load_checkpoint('model.pth')
                    tree_modules = model.update_tree_modules()

                # record the visit to the node
                tree_struct[node_idx]['visited'] = True

                # save the model and tree structures:
                checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules,
                                 data_loader=test_loader,
                                 figname='hist_split_node_{:03d}.png'.format(node_idx))
                checkpoint_msc(tree_struct, records)
                last_node = node_idx

                # global refinement prior to the next growth
                # NOTE: this is an option not included in the paper.
                if args.finetune_during_growth and (criteria == 1 or criteria == 2):
                    print("\n-------------- Global refinement --------------")   
                    model = Tree(tree_struct, tree_modules,
                                 split=False, node_split=last_node,
                                 extend=False, node_extend=last_node,
                                 cuda_on=args.cuda)
                    if args.cuda: 
                        model.cuda()

                    model, tree_modules = optimize_fixed_tree(
                        model, tree_struct,
                        train_loader, valid_loader, test_loader,
                        args.epochs_finetune_node, node_idx,
                    )
        # terminate the tree growth if no split or extend in the final layer
        if not change: break

    # ############### 2: Refinement (finetuning) phase starts #################
    print("\n\n------------------- Fine-tuning the tree --------------------")
    best_valid_accuracy_before = records['valid_best_accuracy']
    model = Tree(tree_struct, tree_modules,
                 split=False,
                 node_split=last_node,
                 child_left=None, child_right=None,
                 extend=False,
                 node_extend=last_node, child_extension=None,
                 cuda_on=args.cuda)
    if args.cuda: 
        model.cuda()

    model, tree_modules = optimize_fixed_tree(model, tree_struct,
                                              train_loader, valid_loader, test_loader,
                                              args.epochs_finetune,
                                              last_node)

    best_valid_accuracy_after = records['valid_best_accuracy']

    # only save if fine-tuning improves validation accuracy
    if best_valid_accuracy_after - best_valid_accuracy_before > 0:
        checkpoint_model('model.pth', struct=tree_struct, modules=tree_modules,
                         data_loader=test_loader,
                         figname='hist_split_node_finetune.png')
    checkpoint_msc(tree_struct, records)
    
    # ############### 3: Final evaluation (soft vs hard) #####################
    # Evaluate final model in both soft and hard inference modes
    model_final = Tree(tree_struct, tree_modules,
                       split=False,
                       cuda_on=args.cuda)
    if args.cuda:
        model_final.cuda()

    print_model_parameter_stats(model_final, title='Final tree model')
    
    evaluate_and_record_final_results(
        model_final, tree_struct,
        train_loader, valid_loader, test_loader,
        start_time=start
    )
    
    # Save the final records with soft/hard results
    checkpoint_msc(tree_struct, records)


# --------------------------- Start growing an ANT! ---------------------------
start = time.time()
grow_ant_nodewise()