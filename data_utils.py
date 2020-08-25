from collections import defaultdict
from collections import deque
from tqdm import tqdm
from types import SimpleNamespace
import logging
import numpy as np
import os
import random
import shutil

from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Coauthor, CoraFull, Planetoid, PPI, Reddit
from torch_geometric.transforms import AddSelfLoops, Compose, NormalizeFeatures, RemoveIsolatedNodes
from torch_geometric.utils import to_undirected
import torch

LOG = logging.getLogger(__name__)
SAMPLE = random.sample
RANDOM = random.random
SHUFFLE = random.shuffle


class NeighborhoodCache(object):
    """
    Store mappings from a node to its neighbors and their distances up to `max_size` entries as a simple queue.
    """
    def __init__(self, max_size):
        self.max_size = max_size
        self.entries = deque()
        self.distances_to_neighbors = {}
        self.neighbors_to_distances = {}

    def get(self, start_node, radius, mode):
        if (start_node, radius, mode) in self.neighbors_to_distances:
            neighbors_to_distances = self.neighbors_to_distances[(start_node, radius, mode)]
            distances_to_neighbors = self.distances_to_neighbors[(start_node, radius, mode)]
            return neighbors_to_distances, distances_to_neighbors

    def put(self, start_node, radius, mode, neighbors_to_distance, distance_to_neighbors):
        if (start_node, radius, mode) not in neighbors_to_distance:
            self.entries.append((start_node, radius, mode))
        if len(self.entries) > self.max_size:
            node_to_delete, radius_to_delete, mode_to_delete = self.entries.popleft()
            del self.neighbors_to_distances[(node_to_delete, radius_to_delete, mode_to_delete)]
            del self.distances_to_neighbors[(node_to_delete, radius_to_delete, mode_to_delete)]
        self.neighbors_to_distances[(start_node, radius, mode)] = neighbors_to_distance
        self.distances_to_neighbors[(start_node, radius, mode)] = distance_to_neighbors


def _get_train_val_test_masks(total_size, y_true, val_fraction, test_fraction, seed):
    """
    Get train, val and test split masks for `total_size` examples with the labels `y_true`. Performs stratified
    splitting over the labels `y_true`. `y_true` is a numpy array.
    """
    indexes = range(total_size)
    indexes_train, indexes_test = train_test_split(indexes, test_size=test_fraction, stratify=y_true, random_state=0)
    indexes_train, indexes_val = train_test_split(indexes_train, test_size=val_fraction, stratify=y_true[indexes_train],
                                                  random_state=seed)
    train_idxs, val_idxs = np.zeros(total_size, dtype=np.bool), np.zeros(total_size, dtype=bool)
    test_idxs = np.zeros(total_size, dtype=np.bool)
    train_idxs[indexes_train] = True
    val_idxs[indexes_val] = True
    test_idxs[indexes_test] = True
    return torch.from_numpy(train_idxs), torch.from_numpy(val_idxs), torch.from_numpy(test_idxs)


def get_small_dataset(dataset_name, normalize_attributes=False, add_self_loops=False, remove_isolated_nodes=False,
                      make_undirected=False, graph_availability=None, seed=0, create_adjacency_lists=True):
    """
    Get the pytorch_geometric.data.Data object associated with the specified dataset name.
    :param dataset_name: str => One of the datasets mentioned below.
    :param normalize_attributes: Whether the attributes for each node should be normalized to sum to 1.
    :param add_self_loops: Add self loops to the input Graph.
    :param remove_isolated_nodes: Remove isolated nodes.
    :param make_undirected: Make the Graph undirected.
    :param graph_availability: Either inductive and transductive. If transductive, all the graph nodes are available
                               during training. Otherwise, only training split nodes are available.
    :param seed: The random seed to use while splitting into train/val/test splits.
    :param create_adjacency_lists: Whether to process and store adjacency lists that can be used for efficient
                                   r-radius neighborhood sampling.
    :return: A pytorch_geometric.data.Data object for that dataset.
    """
    assert dataset_name in {'amazon-computers', 'amazon-photo', 'citeseer', 'coauthor-cs', 'coauthor-physics', 'cora',
                            'cora-full', 'ppi', 'pubmed', 'reddit'}
    assert graph_availability in {'inductive', 'transductive'}

    # Compose transforms that should be applied.
    transforms = []
    if normalize_attributes:
        transforms.append(NormalizeFeatures())
    if remove_isolated_nodes:
        transforms.append(RemoveIsolatedNodes())
    if add_self_loops:
        transforms.append(AddSelfLoops())
    transforms = Compose(transforms) if transforms else None

    # Load the specified dataset and apply transforms.
    root_dir = '/tmp/{dir}'.format(dir=dataset_name)
    processed_dir = os.path.join(root_dir, dataset_name, 'processed')
    # Remove any previously pre-processed data, so pytorch_geometric can pre-process it again.
    if os.path.exists(processed_dir) and os.path.isdir(processed_dir):
        shutil.rmtree(processed_dir)

    data = None

    def split_function(y):
        return _get_train_val_test_masks(y.shape[0], y, 0.2, 0.2, seed)

    if dataset_name in ['citeseer', 'cora', 'pubmed']:
        data = Planetoid(root=root_dir, name=dataset_name, pre_transform=transforms, split='full').data
        if seed != 0:
            data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
        data.graphs = [data]
    elif dataset_name == 'cora-full':
        data = CoraFull(root=root_dir, pre_transform=transforms).data
        data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
        data.graphs = [data]
    elif dataset_name == 'amazon-computers':
        data = Amazon(root=root_dir, name='Computers', pre_transform=transforms).data
        data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
        data.graphs = [data]
    elif dataset_name == 'amazon-photo':
        data = Amazon(root=root_dir, name='Photo', pre_transform=transforms).data
        data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
        data.graphs = [data]
    elif dataset_name == 'coauthor-cs':
        data = Coauthor(root=root_dir, name='CS', pre_transform=transforms).data
        data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
        data.graphs = [data]
    elif dataset_name == 'coauthor-physics':
        data = Coauthor(root=root_dir, name='Physics', pre_transform=transforms).data
        data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
        data.graphs = [data]
    elif dataset_name == 'reddit':
        data = Reddit(root=root_dir, pre_transform=transforms).data
        if seed != 0:
            data.train_mask, data.val_mask, data.test_mask = split_function(data.y.numpy())
        data.graphs = [data]
    elif dataset_name == 'ppi':
        data = SimpleNamespace()
        data.graphs = []
        for split in ['train', 'val', 'test']:
            split_data = PPI(root=root_dir, split=split, pre_transform=transforms)
            x_idxs = split_data.slices['x'].numpy()
            edge_idxs = split_data.slices['edge_index'].numpy()
            split_data = split_data.data
            for x_start, x_end, e_start, e_end in zip(x_idxs, x_idxs[1:], edge_idxs, edge_idxs[1:]):
                graph = Data(split_data.x[x_start:x_end], split_data.edge_index[:, e_start:e_end],
                             y=split_data.y[x_start:x_end])
                graph.num_nodes = int(x_end - x_start)
                graph.split = split
                all_true = torch.ones(graph.num_nodes).bool()
                all_false = torch.zeros(graph.num_nodes).bool()
                graph.train_mask = all_true if split == 'train' else all_false
                graph.val_mask = all_true if split == 'val' else all_false
                graph.test_mask = all_true if split == 'test' else all_false
                data.graphs.append(graph)
        if seed != 0:
            temp_random = random.Random(seed)
            val_graphs = temp_random.sample(range(len(data.graphs)), 2)
            test_candidates = [graph_idx for graph_idx in range(len(data.graphs)) if graph_idx not in val_graphs]
            test_graphs = temp_random.sample(test_candidates, 2)
            for graph_idx, graph in enumerate(data.graphs):
                all_true = torch.ones(graph.num_nodes).bool()
                all_false = torch.zeros(graph.num_nodes).bool()
                graph.split = 'test' if graph_idx in test_graphs else 'val' if graph_idx in val_graphs else 'train'
                graph.train_mask = all_true if graph.split == 'train' else all_false
                graph.val_mask = all_true if graph.split == 'val' else all_false
                graph.test_mask = all_true if graph.split == 'test' else all_false

    if make_undirected:
        for graph in data.graphs:
            graph.edge_index = to_undirected(graph.edge_index, graph.num_nodes)

    LOG.info(f'Downloaded and transformed {len(data.graphs)} graph(s).')

    # Populate adjacency lists for efficient k-neighborhood sampling.
    # Only retain edges coming into a node and reverse the edges for the purpose of adjacency lists.
    LOG.info('Processing adjacency lists and degree information.')

    for graph in data.graphs:
        train_in_degrees = np.zeros(graph.num_nodes, dtype=np.int64)
        val_in_degrees = np.zeros(graph.num_nodes, dtype=np.int64)
        test_in_degrees = np.zeros(graph.num_nodes, dtype=np.int64)
        adjacency_lists = defaultdict(list)
        not_val_test_mask = (~graph.val_mask & ~graph.test_mask).numpy()
        val_mask = graph.val_mask.numpy()
        test_mask = graph.test_mask.numpy()

        if create_adjacency_lists:
            num_edges = graph.edge_index[0].shape[0]
            sources, dests = graph.edge_index[0].numpy(), graph.edge_index[1].numpy()
            for source, dest in tqdm(zip(sources, dests), total=num_edges, leave=False):
                if not_val_test_mask[dest] and not_val_test_mask[source]:
                    train_in_degrees[dest] += 1
                    val_in_degrees[dest] += 1
                elif val_mask[dest] and not test_mask[source]:
                    val_in_degrees[dest] += 1
                test_in_degrees[dest] += 1
                adjacency_lists[dest].append(source)

        graph.adjacency_lists = dict(adjacency_lists)
        graph.train_in_degrees = torch.from_numpy(train_in_degrees).long()
        graph.val_in_degrees = torch.from_numpy(val_in_degrees).long()
        graph.test_in_degrees = torch.from_numpy(test_in_degrees).long()
        if graph_availability == 'transductive':
            graph.train_in_degrees = data.test_in_degrees
            graph.val_in_degrees = data.test_in_degrees

        graph.graph_availability = graph_availability

        # To accumulate any neighborhood perturbations to the graph.
        graph.perturbed_neighborhoods = defaultdict(set)
        graph.added_nodes = defaultdict(set)
        graph.modified_degrees = {}

        # For small datasets, cache the neighborhoods for all nodes for at least 3 different radii queries.
        graph.use_cache = True
        graph.neighborhood_cache = NeighborhoodCache(graph.num_nodes * 3)

        graph.train_mask_original = graph.train_mask
        graph.val_mask_original = graph.val_mask
        graph.test_mask_original = graph.test_mask

        graph.train_mask = torch.ones(graph.num_nodes).bool() & ~graph.val_mask & ~graph.test_mask

    return data


def get_mask_from_mode(graph_dataset, mode):
    """
    Get a mask of nodes that should be visible for the given `mode`. Mode is one of `train`, `val` or `test.
    """
    mask = torch.zeros(graph_dataset.num_nodes).bool()
    if not mode or graph_dataset.graph_availability == 'transductive':
        mask = torch.ones(graph_dataset.num_nodes).bool()
    if mode == 'train':
        # Use only training nodes to discover neighborhoods if training.
        mask = mask | graph_dataset.train_mask
    if mode == 'val':
        # We can use both validation and training nodes to discover neighborhoods during validation.
        mask = mask | graph_dataset.train_mask | graph_dataset.val_mask
    if mode == 'test':
        # We can use all nodes to discover neighborhoods during testing.
        mask = mask | graph_dataset.val_mask | graph_dataset.train_mask | graph_dataset.test_mask
    return mask


def _truncate_neighbors(graph_dataset, neighbors_list, elements_to_distance, max_values, mode):
    if len(neighbors_list) < max_values:
        return neighbors_list

    if mode == 'train':
        in_degrees = graph_dataset.train_in_degrees.numpy()
    elif mode == 'val':
        in_degrees = graph_dataset.val_in_degrees.numpy()
    else:
        in_degrees = graph_dataset.test_in_degrees.numpy()

    prefer_closeness = graph_dataset.prefer_closeness
    neighbors = sorted(
        neighbors_list,
        key=lambda n: (elements_to_distance.get(n, 1) == 0,  # Always include the node itself.
                       # Prefer elements that are closer to the source node.
                       -elements_to_distance.get(n, 0) if prefer_closeness else in_degrees[n],
                       # Break ties by picking the highest degree nodes in the neighborhood.
                       in_degrees[n] if prefer_closeness else -elements_to_distance.get(n, 0),
                       RANDOM() if mode == 'train' else -n),  # Consistently/randomly break ties depending on train/val.
        reverse=True)

    return neighbors[:max_values]


def get_r_hop_nodes(graph_dataset, start_node, radius, max_values, mode):
    """
    Get the nodes reachable within `radius` hops of the `start_node` in the given graph.
    :param graph_dataset: The underlying graph as a pytorch_geometric.data.Data object.
    :param start_node: The integer node id of the source node.
    :param radius: The maximum hops as defined above.
    :param max_values: The maximum number of neighbors to examine for each node.
    :param mode: One of 'train', 'val' or 'test'. All nodes not in the current mode are ignored.
                 If None, all nodes are used.
    :return: dict[int -> int], dict[int -> list[int]] which map nodes to their distance to `start_node` and store all
             nodes reachable that are exactly r minimum hops away from `start_node` for r <= radius respectively.
    """
    # Search existing cache.
    is_perturbed = start_node in graph_dataset.perturbed_neighborhoods or start_node in graph_dataset.added_nodes
    if graph_dataset.use_cache and not is_perturbed and graph_dataset.neighborhood_cache.get(start_node, radius, mode):
        return graph_dataset.neighborhood_cache.get(start_node, radius, mode)

    mask = get_mask_from_mode(graph_dataset, mode)

    # Mappings from a neighbor to its distance to `start_node` and the inverse of the same.
    neighbor_to_distance = {start_node: 0}
    distance_to_neighbor = defaultdict(list, {0: [start_node]})

    removed_nodes = graph_dataset.perturbed_neighborhoods.get(start_node, set())
    added_nodes = graph_dataset.added_nodes.get(start_node, set())

    # Queue for breadth-first search.
    queue = deque([(start_node, 0)])

    while queue:
        node, distance = queue.popleft()
        adjacent_nodes = graph_dataset.adjacency_lists.get(node, set())
        adjacent_nodes = _truncate_neighbors(graph_dataset, adjacent_nodes, {}, max_values, mode)
        if node == start_node:
            adjacent_nodes = set(adjacent_nodes).union(added_nodes).difference(removed_nodes)
        for neighbor in adjacent_nodes:
            if neighbor in neighbor_to_distance or not mask[neighbor]:
                continue
            neighbor_to_distance[neighbor] = distance + 1
            distance_to_neighbor[distance + 1].append(neighbor)
            if distance + 1 < radius:
                queue.append((neighbor, distance + 1))

    # Cache for future use.
    if graph_dataset.use_cache and not is_perturbed:
        graph_dataset.neighborhood_cache.put(start_node, radius, mode, neighbor_to_distance, distance_to_neighbor)
    return neighbor_to_distance, distance_to_neighbor


def _truncate_by_distance(graph_dataset, distances_to_elements, elements_to_distance, max_values, mode):
    neighbors = [elem for elems in distances_to_elements.values() for elem in elems]
    neighbors = _truncate_neighbors(graph_dataset, neighbors, elements_to_distance, max_values, mode)
    neighbors_to_shuffle = neighbors[1:]
    SHUFFLE(neighbors_to_shuffle)
    return [neighbors[0]] + neighbors_to_shuffle


def get_neighbors(graph_dataset, start_nodes, radius, mode, max_neighborhood):
    """
    Get the neighbors corresponding to each node in `start_nodes`.
    :param graph_dataset: The graph as a pytorch_geometric.data.Data object.
    :param start_nodes: A list of nodes, with each element being a node to compute neighbors for.
    :param radius: The r-hop radius of `start_nodes` induced by `graph_dataset` that will be returned as neighbors.
    :param mode: One of 'train', 'val' or 'test' specifying the nodes to be used.
    :param max_neighborhood: The maximum size to truncate a neighborhood to. If a node has more neighbors than this
                             maximum cap, we randomly sample `max_neighborhood` of the nodes.
    :returns neighbors: List[List[int]], with one list per start_node corresponding to the neighbors for the node.
             neighbors_to_distances: List[Dict[int -> int]], one dictionary per node in `start_nodes` containing the
                                   mapping of the neighbor for the node to its distance from the node. Used to encode
                                   the node distances in the model.
             distances_to_neighbors: List[Dict[int -> List[int]]], one dictionary per node in `start_nodes` containing
                                     the mapping from a distance to the nodes at that distance from `start_node`.
                                     Can be seen as an inverse mapping of `neighbors_to_distances`.
    """
    neighbors_to_distances, distances_to_neighbors = zip(
        *[get_r_hop_nodes(graph_dataset, start_node, radius, max_neighborhood, mode) for start_node in start_nodes])

    neighbors = [_truncate_by_distance(graph_dataset, distances_to_neighbors_i, neighbors_to_distance_i,
                                       max_neighborhood, mode)
                 for start_node, distances_to_neighbors_i, neighbors_to_distance_i
                 in zip(start_nodes, distances_to_neighbors, neighbors_to_distances)]

    return neighbors, neighbors_to_distances, distances_to_neighbors
