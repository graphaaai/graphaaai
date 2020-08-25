#!/usr/bin/env python3

import argparse
import logging
import numpy
import os
import random
import signal
import sys
import time

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.adamw import AdamW
import torch

from data_utils import get_small_dataset
from eval_helpers import CROSS_ENTROPY, LOSS_FUNCTIONS
from eval_helpers import log_metrics, raw_output_to_prediction
from fc_network.fc_network import FCNetwork
from fc_network.training_helpers import training_epoch as fc_network_train_epoch
from fc_network.training_helpers import validation_epoch as fc_network_val_epoch
from fc_network.training_helpers import predict as fc_network_predict
from node_encoder.node_encoder import NodeEncoder, SupervisedNodeEncoder
from node_encoder.training_helpers import EarlyStopper, predict
from node_encoder.training_helpers import training_epoch as node_encoder_train_epoch
from node_encoder.training_helpers import training_epoch_end_to_end as end_to_end_train_epoch
from node_encoder.training_helpers import validation_epoch as node_encoder_val_epoch
from node_encoder.training_helpers import validation_epoch_end_to_end as end_to_end_val_epoch

# Set up logging to /var/log.
LOG = logging.getLogger()
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), filename='/var/log/graphaaai.log')

# Add an additional handler to output to stdout.
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(os.environ.get("LOGLEVEL", "INFO"))
LOG.addHandler(_stdout_handler)

# Global flag passed through a signal to indicate that training should be stopped after this epoch.
_STOP_TRAINING = False


def _stop_training_handler(signum, frame):
    global _STOP_TRAINING
    LOG.info('Ok! Stopping after this epoch.')
    _STOP_TRAINING = True


def _init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dataset', type=str.lower, required=True)
    parser.add_argument('--attention_radius', type=int, default=2,
                        help='The r-hop radius to compute attention over.')
    parser.add_argument('--num_encoder_layers', type=int, default=1,
                        help='The number of stacked encoder layers to use.')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                        help='The number of attention heads to use per encoder layer.')
    parser.add_argument('--attention_dim', type=int, default=64,
                        help='Dimensionality to use for the attention space.')
    parser.add_argument('--degree_distance_embed_dim', type=int, default=16,
                        help='The number of dimensions to embed the degree and distance encodings in.')
    parser.add_argument('--output_dim', type=int, default=64,
                        help='Dimensionality to use for the output embeddings.')
    parser.add_argument('--transformer_fc_dim', type=int, default=64,
                        help='Dimensionality to use for the hidden layer of the 2-layer fully-connected NN.')
    parser.add_argument('--positive_radius', type=int, default=2,
                        help='The r-hop radius to sample positives from.')
    parser.add_argument('--negative_radius', type=int, default=5,
                        help='The r-hop radius to sample negatives from.')
    parser.add_argument('--attention_negatives_start_epoch', type=int, default=25,
                        help='The epoch at which current we should start using the current model attention weights '
                             'to sample difficult negatives. Any value > epochs can be given to disable this feature.')
    parser.add_argument('--max_neighbors', type=int, default=50,
                        help='The maximum number of neighbors to use for a node. If a node has more than these many '
                             'neighbors, max_neighbors neighbors are selected according to the logic in data_utils.')
    parser.add_argument('--max_degree', type=int, default=30,
                        help='The maximum degree to encode. Degrees greater than this value are clamped to this value '
                             'and then encoded.')
    parser.add_argument('--loss_margin', type=float, default=0.25,
                        help='The margin value to use in the triplet loss formulation.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--loss_fn', default=CROSS_ENTROPY, choices=LOSS_FUNCTIONS)
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='The probability with which to drop values in various parts of the model.')
    parser.add_argument('--unsupervised_model_patience', type=int, default=20,
                        help='Stop training early if the validation loss does not decrease for these many epochs.')
    parser.add_argument('--supervised_model_patience', type=int, default=10,
                        help='Stop training early if the validation loss does not decrease for these many epochs.')
    parser.add_argument('--progress', action='store_true',
                        help='Print a progress bar for training epoch steps to stdout.')
    parser.add_argument('--model_dir', required=True,
                        help='Path to use to store log files as well as trained models.')
    parser.add_argument('--run_type', required=True, default='both',
                        choices={'unsupervised', 'supervised', 'both', 'end-to-end'},
                        help='Which models to train - unsupervised, supervised, or both. End-to-end trains an '
                             'embedding + classification model in an end-to-end manner using supervised data.')
    parser.add_argument('--supervised_model_dims', default='64', type=lambda x: [int(dim) for dim in x.split(",")],
                        help='A comma-separated list of hidden dimension values to use for the supervised FC model.')
    parser.add_argument('--supervised_model_activation', choices={'sigmoid', 'relu', 'tanh', 'elu'}, default='relu',
                        help='The activation function to use in the supervised FC model.')
    parser.add_argument('--unsupervised_model_file', default=None,
                        help='If provided, the file from which the unsupervised embeddings model should be loaded.')
    parser.add_argument('--beta_start', default=0, type=float,
                        help='The starting value of the coefficient for the perturbed graph loss.')
    parser.add_argument('--beta_end', default=0.5, type=float,
                        help='The ending value of the coefficient for the perturbed graph loss.')
    parser.add_argument('--beta_increment', default=0.1, type=float,
                        help='The value to increment the perturbed loss coefficient beta by in each epoch.')
    parser.add_argument('--perturbed_loss_start_epoch', type=int, default=20,
                        help='The epoch from which the perturbed graph loss term should start.')
    parser.add_argument('--max_perturbations_per_node', type=int, default=5,
                        help='The maximum number of neighbors to remove from the neighborhood of a node while computing'
                             ' the perturbed loss.')
    parser.add_argument('--show_detailed_metrics', action='store_true',
                        help='If enabled, will show degree-wise and neighborhood-size-wise breakdown of metrics.')
    parser.add_argument('--graph_availability', default='inductive', choices={'inductive', 'transductive'},
                        help='If transductive, all the nodes and features are available during training, otherwise only'
                             ' training nodes are seen during training.')
    parser.add_argument('--prefer_closeness', action='store_true',
                        help='If true, prefer nodes closer to the source node while truncating neighborhoods. This is '
                             'recommended for high degree graphs.')
    parser.add_argument('--random_seed', type=int, default=12345,
                        help='The random seed value to use everywhere to attempt to get reproducible results.')
    parser.add_argument('--data_partitioning_seed', type=int, default=0,
                        help='The random seed value to use while splitting the dataset into train/val/test splits.')
    parser.add_argument('--mlp', action='store_true',
                        help='If enabled, train an MLP model as the supervised model directly using node features.')
    return parser


def _add_log_handler(model_dir):
    # Add an additional log handler to output to the model directory.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    _model_log_handler = logging.FileHandler(os.path.join(model_dir, 'train.log'))
    _model_log_handler.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    LOG.addHandler(_model_log_handler)


def _print_args(args):
    LOG.info("-" * 10 + "Arguments:" + "-" * 10)
    for arg in sorted(vars(args)):
        LOG.info("--{arg} {value}".format(arg=arg, value=str(getattr(args, arg))))
    LOG.info("\n" + "-" * 30)


def train_unsupervised(args, graph_dataset, model):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, cooldown=2, factor=0.5)
    early_stopper = EarlyStopper(args.unsupervised_model_patience)

    start_time = time.time()
    LOG.info("Started training unsupervised model.")
    for epoch in range(args.epochs):
        train_loss, train_count = node_encoder_train_epoch(graph_dataset, model, optimizer, epoch, args)
        val_loss, val_count = node_encoder_val_epoch(graph_dataset, model, epoch, args)
        lr_scheduler.step(val_loss)

        epoch_end_time = round(time.time() - start_time, 2)
        LOG.info(f'Epoch {epoch + 1} [{epoch_end_time}s]: Training loss [over {train_count} nodes] = {train_loss:.4f}, '
                 f'Validation loss [over {val_count} nodes] = {val_loss:.4f}')

        if early_stopper.should_stop(model, val_loss) or _STOP_TRAINING:
            break

    LOG.info("Unsupervised training complete.")
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'node_encoder_unsupervised.pt'))


def train_supervised(args, graph_dataset, unsupervised_model, supervised_model):
    # Get the embeddings from the unsupervised model to use as inputs to the supervised model.
    graph = graph_dataset.graphs[0]
    if len(graph_dataset.graphs) > 1:
        x = torch.cat([graph.x for graph in graph_dataset.graphs], dim=0)
        y = torch.cat([graph.y for graph in graph_dataset.graphs], dim=0)
        train_mask = torch.cat([graph.train_mask for graph in graph_dataset.graphs], dim=0)
        val_mask = torch.cat([graph.val_mask for graph in graph_dataset.graphs], dim=0)
        test_mask = torch.cat([graph.test_mask for graph in graph_dataset.graphs], dim=0)
    else:
        x = graph_dataset.x
        y = graph_dataset.y
        train_mask = graph.train_mask
        val_mask = graph.val_mask
        test_mask = graph.test_mask
    if not args.mlp:
        # Predict while still respecting the inductive availability of features.
        assert len(graph_dataset.graphs) == 1, "Only one graph datasets are currently supported."
        x_train = predict(graph_dataset, unsupervised_model, args, mode='train')
        x_val = predict(graph_dataset, unsupervised_model, args, mode='val')
        x_test = predict(graph_dataset, unsupervised_model, args, mode='test')

        x = torch.zeros(graph.num_nodes, args.output_dim, device=x_train.device)
        x[train_mask] = x_train
        x[val_mask] = x_val
        x[test_mask] = x_test
        LOG.info(f'Predicted {x.shape} dimensional embeddings from the unsupervised model.')

    # Train on all nodes that are neither validation nor test nodes.
    optimizer = AdamW(supervised_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=min(25, args.supervised_model_patience // 2), cooldown=10,
                                     factor=0.5)
    early_stopper = EarlyStopper(args.supervised_model_patience)

    epoch = 0
    start_time = time.time()
    LOG.info(f"Started training supervised model.")
    while True:
        train_loss = fc_network_train_epoch(x, y, train_mask, supervised_model, optimizer, epoch, args)
        val_loss = fc_network_val_epoch(x, y, val_mask, supervised_model, epoch, args)
        lr_scheduler.step(val_loss)

        epoch_end_time = round(time.time() - start_time, 2)
        LOG.info(f'Epoch {epoch + 1} [{epoch_end_time}s]: '
                 f'Training loss = {train_loss:.4f}, Validation loss = {val_loss:.4f}')

        if early_stopper.should_stop(supervised_model, val_loss) or _STOP_TRAINING:
            break

        epoch += 1

    LOG.info("Supervised training complete.")
    torch.save(supervised_model.state_dict(), os.path.join(args.model_dir, 'node_encoder_supervised.pt'))

    # Compute metrics.
    val_y_pred = fc_network_predict(x, val_mask, supervised_model, args)
    val_y_pred = raw_output_to_prediction(val_y_pred, args.loss_fn).cpu().numpy()
    val_y_true = y[val_mask].cpu().numpy()
    test_y_pred = fc_network_predict(x, test_mask, supervised_model, args)
    test_y_pred = raw_output_to_prediction(test_y_pred, args.loss_fn).cpu().numpy()
    test_y_true = y[test_mask].cpu().numpy()
    log_metrics(graph_dataset, val_y_true, val_y_pred, test_y_true, test_y_pred, args.show_detailed_metrics,
                args.attention_radius, args.max_neighbors)


def train_end_to_end(args, graph_dataset, model):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer, patience=5, cooldown=2, factor=0.5)
    early_stopper = EarlyStopper(args.supervised_model_patience)

    epoch = 0
    start_time = time.time()
    LOG.info('Started training supervised model.')
    while True:
        train_loss = end_to_end_train_epoch(graph_dataset, model, optimizer, args, epoch)
        val_loss, val_acc, val_f1 = end_to_end_val_epoch(graph_dataset, model, args)
        lr_scheduler.step(val_loss)

        epoch_end_time = round(time.time() - start_time, 2)
        LOG.info(f'Epoch {epoch + 1} [{epoch_end_time}s]: '
                 f'Training loss = {train_loss:.4f}, Validation loss = {val_loss:.4f}, '
                 f'Val accuracy = {val_acc:.4f}, Val f1 = {val_f1:.4f}')
        epoch += 1
        if early_stopper.should_stop(model, val_loss) or _STOP_TRAINING:
            break

    # Compute metrics.
    val_y_pred = predict(graph_dataset, model, args, mode='val')
    val_y_pred = raw_output_to_prediction(val_y_pred, args.loss_fn)
    val_y_true = [label for graph in graph_dataset.graphs for label in graph.y[graph.val_mask].cpu().numpy()]
    test_y_pred = predict(graph_dataset, model, args, mode='test')
    test_y_pred = raw_output_to_prediction(test_y_pred, args.loss_fn)
    test_y_true = [label for graph in graph_dataset.graphs for label in graph.y[graph.test_mask].cpu().numpy()]
    log_metrics(graph_dataset, val_y_true, val_y_pred.cpu().numpy(), test_y_true,
                test_y_pred.cpu().numpy(), args.show_detailed_metrics, args.attention_radius, args.max_neighbors)

    # Save trained end-to-end model.
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'node_encoder_end_to_end.pt'))


if __name__ == "__main__":
    # Parse and store program args.
    args = _init_parser().parse_args()
    _add_log_handler(args.model_dir)
    _print_args(args)

    # Register handler for SIGQUIT.
    signal.signal(signal.SIGQUIT, _stop_training_handler)

    # For reproducibility of results.
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    graph_dataset = get_small_dataset(args.graph_dataset, add_self_loops=True, remove_isolated_nodes=True,
                                      make_undirected=True, graph_availability=args.graph_availability,
                                      seed=args.data_partitioning_seed, create_adjacency_lists=not args.mlp)
    LOG.info(f"Loaded {len(graph_dataset.graphs)} graph(s) into memory with "
             f"{sum([graph.num_nodes for graph in graph_dataset.graphs])} total nodes and "
             f"{sum([graph.edge_index[0].shape[0] for graph in graph_dataset.graphs])} total edges.")
    if args.graph_dataset != 'ppi':
        LOG.info(f'Train nodes: {graph_dataset.train_mask.sum()}, Val nodes: {graph_dataset.val_mask.sum()}, '
                 f'Test nodes: {graph_dataset.test_mask.sum()}.')

    # Compute properties required for the model.
    # Since we're using a fixed attention radius, positive_radius and negative_radius, we have an upper bound on the
    # maximum number of unique distances between pairs of nodes that we need to encode (embed).
    max_distance = max([args.attention_radius, args.positive_radius, args.negative_radius])
    max_degree = args.max_degree
    # Get the number of input features.
    num_features = graph_dataset.graphs[0].num_node_features
    # Get the number of output classes.
    num_classes = torch.unique(graph_dataset.graphs[0].y).size()[0]
    if args.loss_fn == 'binary_cross_entropy':
        num_classes = graph_dataset.graphs[0].y.shape[1]
    graph_dataset.num_classes = num_classes
    for graph in graph_dataset.graphs:
        graph.prefer_closeness = args.prefer_closeness

    unsupervised_model = None
    if args.run_type in ['unsupervised', 'supervised', 'both']:
        unsupervised_model = NodeEncoder(args.num_encoder_layers, max_degree, max_distance, args.num_attention_heads,
                                         num_features, args.output_dim, args.attention_dim,
                                         args.degree_distance_embed_dim, args.transformer_fc_dim,
                                         p_dropout=args.dropout)
        LOG.info("Created unsupervised model.")
        if args.unsupervised_model_file:
            unsupervised_model.load_state_dict(torch.load(args.unsupervised_model_file))
            LOG.info(f'Loaded model weights from {args.unsupervised_model_file}.')

    supervised_model = None
    if args.run_type in ['supervised', 'both']:
        in_dim = args.output_dim
        out_dim = num_classes
        assert args.supervised_model_dims is not None
        supervised_model = FCNetwork(in_dim, out_dim, args.supervised_model_dims, args.supervised_model_activation,
                                     dropout=args.dropout)
        LOG.info("Created supervised model.")

    end_to_end_model = None
    if args.run_type == 'end-to-end':
        end_to_end_model = SupervisedNodeEncoder(args.num_encoder_layers, max_degree, max_distance,
                                                 args.num_attention_heads, num_features, args.output_dim,
                                                 args.attention_dim, args.degree_distance_embed_dim,
                                                 args.transformer_fc_dim, num_classes, args.supervised_model_dims,
                                                 args.supervised_model_activation, dropout=args.dropout)
        LOG.info('Created end-to-end supervised model.')
        if args.unsupervised_model_file:
            end_to_end_model.node_encoder.load_state_dict(torch.load(args.unsupervised_model_file))
            LOG.info(f'Loaded model weights from {args.unsupervised_model_file}.')

    # Move models to default GPU if necessary.
    if args.cuda:
        if unsupervised_model:
            unsupervised_model = unsupervised_model.cuda()
        if supervised_model:
            supervised_model = supervised_model.cuda()
        if end_to_end_model:
            end_to_end_model = end_to_end_model.cuda()
        for graph in graph_dataset.graphs:
            graph.x = graph.x.cuda()
            graph.y = graph.y.cuda()

    LOG.info('-' * 20)
    LOG.info('Press "Ctrl + \\" at anytime to stop training at the end of the current epoch.')
    LOG.info('-' * 20)

    if args.run_type in ['unsupervised', 'both']:
        train_unsupervised(args, graph_dataset, unsupervised_model)
    if args.run_type in ['supervised', 'both']:
        train_supervised(args, graph_dataset, unsupervised_model, supervised_model)
    if args.run_type == 'end-to-end':
        train_end_to_end(args, graph_dataset, end_to_end_model)
