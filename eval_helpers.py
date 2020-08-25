import json
import logging

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

from data_utils import get_neighbors

LOG = logging.getLogger(__name__)

CROSS_ENTROPY = 'cross_entropy'
BINARY_CROSS_ENTROPY = 'binary_cross_entropy'
MEAN_ABSOLUTE_ERROR = 'mean_absolute_error'
LOSS_FUNCTIONS = {CROSS_ENTROPY, BINARY_CROSS_ENTROPY, MEAN_ABSOLUTE_ERROR}


def raw_output_to_prediction(y, loss_fn):
    """
    Contains the logic to convert the raw model output to a prediction for an instance. This is useful for evaluating
    metrics other than the loss (example accuracy).
    :param y: The raw model output, one row per example.
    :param loss_fn: The loss function being used while training.
    """
    if loss_fn == CROSS_ENTROPY:
        return y.max(dim=1)[1]
    elif loss_fn == BINARY_CROSS_ENTROPY:
        return (y >= 0).long()
    elif loss_fn == MEAN_ABSOLUTE_ERROR:
        return y
    raise ValueError(f'Unexpected loss function {loss_fn}.')


def degree_wise_metrics(graph_dataset, y_true, y_pred, mask, mode):
    """
    TODO: Fix this method for multi-graph datasets.
    Break down of F1 score and Accuracy across the various degrees of nodes induced by `mask`.
    :param graph_dataset: A graph represented as a pytorch_geometric.data.Data object.
    :param y_true: Ground truth labels.
    :param y_pred: Model predictions.
    :param mask: A torch bool tensor mask with `True` for indexes of nodes that should be included in the eval.
    :param mode: One of `train`, `val` or `test` specifying the view of the graph to use.
    :return: (Dict[degree -> (f1_score, num_samples)], Dict[degree -> (accuracy, num_samples)])
    """
    assert mode in {'train', 'val', 'test'}
    in_degrees = graph_dataset.train_in_degrees
    if mode == 'val':
        in_degrees = graph_dataset.val_in_degrees
    elif mode == 'test':
        in_degrees = graph_dataset.test_in_degrees
    degrees = in_degrees[mask].numpy()
    f1_scores = {}
    accuracies = {}
    for degree in np.unique(degrees):
        degree_idxs = degrees == degree
        f1_scores[int(degree)] =\
            f'F1-score: {round(f1_score(y_true[degree_idxs], y_pred[degree_idxs], average="micro"), 4)}. '\
            f'Num samples = {degree_idxs.sum()}'
        accuracies[int(degree)] =\
            f'Accuracy: {round(accuracy_score(y_true[degree_idxs], y_pred[degree_idxs]), 4)}. '\
            f'Num samples = {degree_idxs.sum()}'

    return f1_scores, accuracies


def neighborhood_size_wise_metrics(graph_dataset, y_true, y_pred, mask, mode, radius, max_neighbors):
    """
    TODO: Fix this method for multi-graph datasets.
    Break down of F1 score and Accuracy across the various neighborhood sizes at the given r-hop radius for the nodes.
    :param graph_dataset: A graph represented as a pytorch_geometric.data.Data object.
    :param y_true: Ground truth labels.
    :param y_pred: Model predictions.
    :param mask: A torch bool tensor mask with `True` for indexes of nodes that should be included in the eval.
    :param mode: One of `train`, `val` or `test` that decides which nodes are visible during graph traversal.
    :param radius: The r-hop radius at which neighborhoods sizes should be computed.
    :param max_neighbors: Truncate neighborhoods beyond this size by randomly sampling these many nodes.
    :return: (Dict[size -> (f1_score, num_samples)], Dict[size -> (accuracy, num_samples)]), where size is the
             neighborhood size at `radius`.
    """
    neighborhood_sizes = [len(get_neighbors(graph_dataset, [node], radius, mode, max_neighbors)[0][0])
                          for node in np.arange(graph_dataset.num_nodes)[mask.numpy()]]

    f1_scores = {}
    accuracies = {}
    for n_size in np.unique(neighborhood_sizes):
        size_idxs = neighborhood_sizes == n_size
        f1_scores[int(n_size)] = \
            f'F1-score: {round(f1_score(y_true[size_idxs], y_pred[size_idxs], average="micro"), 4)}. ' \
            f'Num samples = {size_idxs.sum()}'
        accuracies[int(n_size)] = \
            f'Accuracy: {round(accuracy_score(y_true[size_idxs], y_pred[size_idxs]), 4)}. ' \
            f'Num samples = {size_idxs.sum()}'

    return f1_scores, accuracies


def log_metrics(graph_dataset, val_y_true, val_y_pred, test_y_true, test_y_pred, show_detailed_metrics,
                attention_radius, max_neighbors):
    """
    Log the validation and testing accuracy and F1 metrics. If `show_detailed_metrics` is true, also displays the
    metrics stratified by degree and neighborhood sizes.
    """
    # Compute validation metrics.
    val_f1_score = f1_score(val_y_true, val_y_pred, average='micro')
    val_accuracy_score = accuracy_score(val_y_true, val_y_pred)

    LOG.info(f'[Val] Micro-averaged F1-score = {val_f1_score:.4f}')
    LOG.info(f'[Val] Accuracy = {val_accuracy_score:.4f}')

    if show_detailed_metrics:
        val_f1_scores_degrees, val_accuracy_scores_degrees = degree_wise_metrics(
            graph_dataset, val_y_true, val_y_pred, graph_dataset.val_mask, 'val')
        val_f1_scores_nsize, val_accuracy_scores_nsize = neighborhood_size_wise_metrics(
            graph_dataset, val_y_true, val_y_pred, graph_dataset.val_mask, 'val', attention_radius, max_neighbors)
        LOG.info(f'[Val] Degree-wise F1-scores = {json.dumps(val_f1_scores_degrees, indent=2)}')
        LOG.info(f'[Val] Degree-wise Accuracies = {json.dumps(val_accuracy_scores_degrees, indent=2)}')
        LOG.info(f'[Val] Neighborhood size-wise F1-scores = {json.dumps(val_f1_scores_nsize, indent=2)}')
        LOG.info(f'[Val] Neighborhood size-wise Accuracies = {json.dumps(val_accuracy_scores_nsize, indent=2)}')

    # Compute test metrics.
    test_f1_score = f1_score(test_y_true, test_y_pred, average='micro')
    test_accuracy_score = accuracy_score(test_y_true, test_y_pred)

    LOG.info(f'[Test] Micro-averaged F1-score = {test_f1_score:.4f}')
    LOG.info(f'[Test] Accuracy = {test_accuracy_score:.4f}')

    if show_detailed_metrics:
        test_f1_scores_degrees, test_accuracy_scores_degrees = degree_wise_metrics(
            graph_dataset, test_y_true, test_y_pred, graph_dataset.test_mask, 'test')
        test_f1_scores_nsize, test_accuracy_scores_nsize = neighborhood_size_wise_metrics(
            graph_dataset, test_y_true, test_y_pred, graph_dataset.test_mask, 'test', attention_radius, max_neighbors)
        LOG.info(f'[Test] Degree-wise F1-scores = {json.dumps(test_f1_scores_degrees, indent=2)}')
        LOG.info(f'[Test] Degree-wise Accuracies = {json.dumps(test_accuracy_scores_degrees, indent=2)}')
        LOG.info(f'[Test] Neighborhood size-wise F1-scores = {json.dumps(test_f1_scores_nsize, indent=2)}')
        LOG.info(f'[Test] Neighborhood size-wise Accuracies = {json.dumps(test_accuracy_scores_nsize, indent=2)}')
