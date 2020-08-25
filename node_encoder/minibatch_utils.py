"""
Utility functions to prepare minibatches for the NodeEncoder model.
"""
from collections import defaultdict
from itertools import chain, islice
import logging
import random

from numpy.random import multinomial
from torch.nn.utils.rnn import pad_sequence
import torch

from data_utils import get_mask_from_mode, get_neighbors, get_r_hop_nodes

LOG = logging.getLogger(__name__)
SAMPLE = random.sample
RANDINT = random.randint


def get_mask(graph_dataset, mode):
    """
    Get the mask corresponding to the training data split.
    :param graph_dataset: The Graph represented as a pytorch_geometric.data.Data object.
    :param mode: One of 'train', 'val' or 'test'. If None, all nodes are used.
    :return: A boolean tensor with True for node indices which belong to the split.
    """
    if not mode:
        return torch.ones(graph_dataset.num_nodes).bool()
    if mode == 'train':
        return graph_dataset.train_mask
    if mode == 'val':
        return graph_dataset.val_mask
    if mode == 'test':
        return graph_dataset.test_mask


def minibatch_generator(graph_dataset, batch_size, attention_radius, positive_radius, negative_radius,
                        use_attention_negatives=True, model=None, shuffle=True, mode=None, is_val=False,
                        max_neighbors=30):
    """
    Create minibatches of nodes from the given graph.
    :param graph_dataset: The Graph represented as a pytorch_geometric.data.Data object.
    :param batch_size: The expected minibatch size.
    :param attention_radius: An integer value specifying the neighborhood radius within which nodes should be attended.
    :param positive_radius: An integer value specifying the neighborhood radius from which positives should be picked.
    :param negative_radius: An integer value specifying the neighborhood radius from which negatives should be picked.
                            More specifically negative nodes are sampled from the region between positive_radius and
                            negative_radius.
    :param use_attention_negatives: When True, negatives are select based on the attention scores of nodes in the
                                    negative_radius - positive_radius region.
    :param model: A `NodeEncoder` model. Only needs to be provided if `use_attention_negatives` is True. Used to compute
                  the current attention values to sample negatives.
    :param shuffle: Whether to shuffle the nodes before creating minibatches.
    :param mode: One of 'train', 'val' or 'test' specifying the nodes to be used in the computations. If none, all nodes
                 are used.
    :param is_val: Whether the current run is for training or validation. If validation, a consistent positive and
                   negative are picked each time, so that loss values are comparable across multiple epochs.
    :param max_neighbors: The maximum number of neighbors to examine per node.
    :yields: Tuples of node batch inputs, positive batch inputs and negative batch inputs so the triplet loss can be
             calculated for corresponding elements in the batch.

    WARNING: This is currently quite CPU intensive due to multiple local graph traversals to compute the neighborhoods
             of the (1) a random base node, (2) a random positive candidate for (1), and (3) a random negative candidate
             for (1). Only recommended for small toy example graphs.
    """
    assert batch_size > 0
    for graph in graph_dataset.graphs:
        # Set all the properties of the current graph in `graph_dataset`.
        graph_dataset.active_graph = graph

        # Proceed with handling graph_dataset like a single, independent graph.
        split_mask = get_mask(graph_dataset.active_graph, mode)
        nodes = torch.arange(graph_dataset.active_graph.num_nodes)[split_mask]
        if nodes.shape[0] == 0:
            continue
        if shuffle:
            nodes = nodes[torch.randperm(nodes.shape[0])]
        partitioned_nodes = torch.split(nodes, batch_size)
        for nodes_minibatch in partitioned_nodes:
            nodes_minibatch = nodes_minibatch.numpy()
            nodes, positives, negatives = get_triplets(graph_dataset.active_graph, nodes_minibatch, attention_radius,
                                                       positive_radius, negative_radius, mode, use_attention_negatives,
                                                       model, max_neighbors, is_val=is_val)

            # Don't yield minibatches when we can't generate positives/negatives for any nodes in it.
            if not nodes:
                continue

            yield nodes, positives, negatives


def get_triplets(graph_dataset, nodes_minibatch, attention_radius, positive_radius, negative_radius, mode,
                 use_attention_negatives, model, max_neighbors, num_positives=2, num_negatives=2, is_val=False):
    """
    See the corresponding arguments of `minibatch_generator`.
    :return: (List[int], List[int], List[int]) all of equal length specifying the nodes, positive and negatives
             to form triplets out of.
    """
    max_radius = max([attention_radius, positive_radius, negative_radius])
    nodes, positives, = [], []
    node_neighbors_to_distances = []
    node_distances_to_neighbors = []
    _, neighbors_to_rs, rs_to_neighbors = get_neighbors(graph_dataset, nodes_minibatch, max_radius, mode, max_neighbors)
    for idx, node in enumerate(nodes_minibatch):
        # Get positive candidates within positive_radius distance.
        positive_rs_to_neighbors = {r: elems for r, elems in rs_to_neighbors[idx].items() if 1 <= r <= positive_radius}
        positive_candidate_counts = [len(radius_candidates)
                                     for radius, radius_candidates in sorted(positive_rs_to_neighbors.items())]
        total_candidates = sum(positive_candidate_counts)
        if total_candidates == 0:
            LOG.debug(f'Node {node}: Found no positives. Ignoring {node} while training.')
            continue

        if is_val:
            positive_candidates = list(islice(chain.from_iterable(positive_rs_to_neighbors.values()), num_positives))
        else:
            normalized_candidate_counts = [count / float(total_candidates) for count in positive_candidate_counts]
            sampled_radii = multinomial(min(num_positives, total_candidates), normalized_candidate_counts)
            positive_candidates = []
            for radius, elems_count in zip(range(1, 1 + len(sampled_radii)), sampled_radii):
                positive_candidates.extend(SAMPLE(positive_rs_to_neighbors[radius],
                                                  min(elems_count, len(positive_rs_to_neighbors[radius]))))

        this_num_positives = len(positive_candidates)
        nodes.extend([node] * this_num_positives)
        positives.extend(positive_candidates)
        node_neighbors_to_distances.extend([neighbors_to_rs[idx]] * this_num_positives)
        node_distances_to_neighbors.extend([rs_to_neighbors[idx]] * this_num_positives)

    if use_attention_negatives and not is_val:
        assert model is not None
        # We should sample negatives based on the current attention values of the model.
        nodes, positives, negatives = _get_attention_negative_triplets(
            graph_dataset, model, nodes, positives, num_negatives, max_neighbors=max_neighbors,
            negative_radius=negative_radius, neighbors_to_distances=node_neighbors_to_distances, mode=mode)
    else:
        nodes, positives, negatives = _get_standard_negative_triplets(
            graph_dataset, nodes, positives, num_negatives, node_neighbors_to_distances, node_distances_to_neighbors,
            max_neighbors, mode, is_val)

    return nodes, positives, negatives


def _get_random_negative_nodes(graph_dataset, node, num_negatives, mask, neighbor_to_distances, positive_distance):
    """
    Sampling random nodes from the graph belonging to the same split (`mask` True) and not belonging to the given
    neighborhood of the source node.
    """
    candidates = torch.arange(graph_dataset.num_nodes)[mask]
    candidates = candidates[torch.randperm(candidates.shape[0])]
    negatives = []
    for candidate in candidates.numpy():
        if candidate == node or neighbor_to_distances.get(candidate, positive_distance + 1) <= positive_distance:
            continue
        if candidate in negatives:
            continue
        negatives.append(candidate)
        if len(negatives) == num_negatives:
            break
    return negatives


def _get_standard_negative_triplets(graph_dataset, nodes, positives, num_negatives, neighbors_to_distances,
                                    distances_to_neighbors, max_neighbors, mode, is_val):
    """
    Get negatives for each (node, positive_node) pair by randomly and uniformly sampling nodes which are farther away
    from a given node than the corresponding positive node.
    :return: (List[int], List[int], List[int]) representing nodes, positives and negatives respectively.
    """
    filtered_nodes, filtered_positives, filtered_negatives = [], [], []
    for idx, (node, pos_candidate) in enumerate(zip(nodes, positives)):
        pos_distance = neighbors_to_distances[idx][pos_candidate]
        negative_rs_to_neighbors = {r: elems for r, elems in distances_to_neighbors[idx].items() if r > pos_distance}
        negative_candidate_counts = [len(radius_candidates)
                                     for radius, radius_candidates in sorted(negative_rs_to_neighbors.items())]
        total_candidates = sum(negative_candidate_counts)
        if total_candidates == 0:
            LOG.debug(f'Sampling random nodes as negative candidates for {node}, {pos_candidate}.')
            negative_candidates = _get_random_negative_nodes(graph_dataset, node, num_negatives,
                                                             get_mask_from_mode(graph_dataset, mode),
                                                             neighbors_to_distances[idx], pos_distance)
        elif is_val:
            negative_candidates = list(islice(chain.from_iterable(negative_rs_to_neighbors.values()), num_negatives))
        else:
            normalized_candidate_counts = [count / float(total_candidates) for count in negative_candidate_counts]
            sampled_radii = multinomial(min(num_negatives, total_candidates), normalized_candidate_counts)
            negative_candidates = []
            start_radius, end_radius = pos_distance + 1, pos_distance + 1 + len(sampled_radii)
            for radius, elems_count in zip(range(start_radius, end_radius), sampled_radii):
                negative_candidates.extend(SAMPLE(negative_rs_to_neighbors[radius],
                                                  min(elems_count, len(negative_rs_to_neighbors[radius]))))

        for neg in negative_candidates:
            filtered_nodes.append(node)
            filtered_positives.append(pos_candidate)
            filtered_negatives.append(neg)

    return filtered_nodes, filtered_positives, filtered_negatives


def _get_attention_negative_triplets(graph_dataset, model, nodes, positives, num_negatives, max_neighbors,
                                     neighbors_to_distances, negative_radius, mode):
    """
    Computes up to `num_negatives` negative nodes for each node in `start_nodes` that can be used as negative examples
    for the node. Picks the nodes with the top attention weights induced by the model.
    :param graph_dataset: The underlying graph as a pytorch_geometric.data.Data object.
    :param model: A `NodeEncoder` model object used to compute the attention scores.
    :param nodes: A LongTensor of start nodes for which negative examples should be computed.
    :param positives: A LongTensor of positive nodes corresponding to each of the `start_nodes`.
    :param num_negatives: The number of negatives to sample per (node, positive_node) pair.
    :param max_neighbors: The maximum number of neighbors to examine per node.
    :param neighbors_to_distances: List[Dict[int -> int]], one dictionary per node in `start_nodes` containing the
                                   mapping of the neighbor for the node to its distance from the node. Used to encode
                                   the node distances in the model.
    :param negative_radius: The maximum radius at which negatives should be sampled.
    :param mode: One of 'train', 'val' or 'test' specifying the nodes to be used.
    """
    filtered_nodes, filtered_positives, filtered_negatives = [], [], []
    mask = get_mask_from_mode(graph_dataset, mode)
    # Disable gradient tracking while computing the attention weights here. This is not strictly necessary, as gradients
    # can't flow from this computation to the loss, but explicitly setting this is more efficient.
    with torch.no_grad():
        # (batch, n_heads, num_neighbors)
        neighbors_attn, attention_weights = model.get_neighbor_attentions(graph_dataset, nodes, max_neighbors,
                                                                          negative_radius, mode)
        neighbors_attn = [torch.LongTensor(neighbors) for neighbors in neighbors_attn]
        neighbors_attn = pad_sequence(neighbors_attn, batch_first=True, padding_value=-1)
        # For now, take the attention value for a neighbor as the max across all attention heads.
        attention_weights, _ = attention_weights.max(dim=1)  # (batch, num_neighbors)
        _, attention_weights = attention_weights.sort(descending=True, dim=1)
        for batch_idx, (node, positive) in enumerate(zip(nodes, positives)):
            negative_candidates = neighbors_attn[batch_idx][attention_weights[batch_idx]].numpy()
            this_num_negatives = 0
            pos_distance = neighbors_to_distances[batch_idx][positive]
            for neg in negative_candidates:
                if neg < 0:
                    break
                neg_distance = neighbors_to_distances[batch_idx][neg]
                if mask[neg] and neg_distance > pos_distance:
                    filtered_nodes.append(node)
                    filtered_positives.append(positive)
                    filtered_negatives.append(neg)
                    this_num_negatives += 1
                    if this_num_negatives >= num_negatives:
                        break

            # Sample random negatives if we are unable to find smart ones in our neighborhood.
            if this_num_negatives == 0:
                LOG.debug(f'Sampling random nodes as negative candidates for {node}, {positive}.')
                negative_candidates = _get_random_negative_nodes(graph_dataset, node, num_negatives, mask,
                                                                 neighbors_to_distances[batch_idx], pos_distance)
                for neg in negative_candidates:
                    filtered_nodes.append(node)
                    filtered_positives.append(positive)
                    filtered_negatives.append(neg)

        return filtered_nodes, filtered_positives, filtered_negatives


def perturb_graph(graph_dataset, model, nodes, max_neighbors, mode, attention_radius, max_perturbations_per_node=1):
    """
    Perturb a graph by removing the node with the highest attention from the neighborhood of the source node.
    Modifies the graph in-place and stores the perturbations so they can be undone in the `unperturb_graph` method of
    this module.
    """
    with torch.no_grad():
        unique_node_idxs = {node: idx for idx, node in enumerate(nodes)}

        # (batch, n_heads, num_neighbors)
        neighbors_attn, attention_weights = model.get_neighbor_attentions(graph_dataset, nodes, max_neighbors, 1, mode)
        neighbors_attn_padded = [torch.LongTensor(neighbors) for neighbors in neighbors_attn]
        neighbors_attn_padded = pad_sequence(neighbors_attn_padded, batch_first=True, padding_value=-1)
        # For now, take the attention value for a neighbor as the maximum across all attention heads.
        attention_weights, _ = attention_weights.max(dim=1)  # (batch, num_neighbors)
        _, attention_weight_idxs = attention_weights.sort(descending=True, dim=1)
        for node, batch_idx in unique_node_idxs.items():
            perturbations_per_node = RANDINT(1, max_perturbations_per_node)
            # Go through the neighbors for this node in descending order of attention values.
            sorted_attention_nodes = neighbors_attn_padded[batch_idx][attention_weight_idxs[batch_idx]].numpy()
            removed_nodes = []
            for attention_node in sorted_attention_nodes:
                if attention_node < 0 or len(removed_nodes) >= perturbations_per_node:
                    break
                if attention_node == node:
                    continue
                if len(removed_nodes) < perturbations_per_node:
                    removed_nodes.append(attention_node)

            added_nodes = []
            # Add new nodes 50% of the times, and add a random number of nodes.
            num_to_add = RANDINT(0, 1) * RANDINT(1, perturbations_per_node)
            if num_to_add > 0:
                neighborhood, _ = get_r_hop_nodes(graph_dataset, node, attention_radius, max_neighbors, mode)
                mask = get_mask_from_mode(graph_dataset, mode)
                while len(added_nodes) < num_to_add:
                    to_add = RANDINT(0, graph_dataset.num_nodes - 1)
                    if to_add == node or not mask[to_add] or to_add in added_nodes or to_add in neighborhood:
                        continue
                    added_nodes.append(to_add)

            graph_dataset.perturbed_neighborhoods[node].update(removed_nodes)
            graph_dataset.added_nodes[node].update(added_nodes)
            graph_dataset.modified_degrees[node] = graph_dataset.train_in_degrees.clone()
            graph_dataset.modified_degrees[node][node] += len(added_nodes) - len(removed_nodes)
            for removed_node in removed_nodes:
                graph_dataset.modified_degrees[node][removed_node] -= 1
            for added_node in added_nodes:
                graph_dataset.modified_degrees[node][added_node] += 1


def unperturb_graph(graph_dataset):
    """
    Unperturb a graph that was perturbed using the `perturb_graph` method of this module. Modifies the graph in-place.
    :param graph_dataset: The underlying graph dataset as a pytorch_geometric.data.Data object.
    """
    graph_dataset.perturbed_neighborhoods = defaultdict(set)
    graph_dataset.added_nodes = defaultdict(set)
    graph_dataset.modified_degrees = {}


def prediction_minibatch_generator(graph_dataset, batch_size, mode, shuffle=False, labels=False):
    """
    Return minibatches of data in the format that expected by the `NodeEncoder` model.
    :param graph_dataset: The underlying graph dataset as a pytorch_geometric.data.Data object.
    :param batch_size: The batch size to use.
    :param mode: One of 'train', 'val' or 'test'.
    :param shuffle: Whether the dataset should be shuffled.
    :param labels: Also return the true labels.
    """
    assert batch_size > 0
    for graph in graph_dataset.graphs:
        # Set all the properties of the current graph in `graph_dataset`.
        graph_dataset.active_graph = graph

        # Proceed with handling graph_dataset like a single, independent graph.
        nodes = torch.arange(graph_dataset.active_graph.num_nodes)[get_mask(graph_dataset.active_graph, mode)]
        if nodes.shape[0] == 0:
            continue
        if shuffle:
            nodes = nodes[torch.randperm(nodes.shape[0])]
        partitioned_nodes = torch.split(nodes, batch_size)
        for nodes_minibatch in partitioned_nodes:
            if labels:
                yield graph_dataset.active_graph.y[nodes_minibatch], nodes_minibatch.numpy()
            else:
                yield nodes_minibatch.numpy()
