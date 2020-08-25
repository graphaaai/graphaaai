import logging

from torch.nn import Dropout, Embedding, LayerNorm, Linear
from torch.nn.functional import relu, softmax
from torch.nn.modules import Module
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch

from data_utils import get_neighbors
from fc_network.fc_network import FCNetwork

LOG = logging.getLogger(__name__)


class DegreeEmbedding(Module):
    """
    Defines the embeddings to encode the degrees of nodes in a graph.

    Args:
        max_degree: The maximum degree of any node in the graph.
        embedding_dim: Dimension of the degree embedding.
        norm_p: The L-norm of the embeddings to normalize by, example L2 norm, L1 norm or L infinity norm.
    """
    def __init__(self, max_degree, embedding_dim, norm_p=float('inf')):
        super(DegreeEmbedding, self).__init__()
        self.max_degree = max_degree
        self.embedding = Embedding(max_degree + 1, embedding_dim, max_norm=1.0, norm_type=norm_p, padding_idx=0)

    def forward(self, x):
        x = x.clamp_max(self.max_degree)
        return self.embedding(x)


class DistanceEmbedding(Module):
    """
    Defines the embeddings to encode the distances between any two given nodes in the graph.

    Args:
        diameter: The diameter of the underlying graph.
        embedding_dim: Dimensions of the distance embeddings.
        norm_p: The L-norm of the embeddings to normalize by, example L2 norm, L1 norm or L-infinity norm.
    """
    def __init__(self, diameter, embedding_dim, norm_p=float('inf')):
        super(DistanceEmbedding, self).__init__()
        self.diameter = diameter
        self.embedding = Embedding(diameter + 1, embedding_dim, max_norm=1.0, norm_type=norm_p)

    def forward(self, x):
        x = x.clamp_max(self.diameter)
        return self.embedding(x)


class NeighborAttention(Module):
    """
    Implements multi-headed attention across the neighbours of a graph node.

    Args:
        n_heads: Number of attention heads.
        d_x: Input dimension dx for node attribute vector, which is also the output dimension.
        d_att: Dimension of the attention space per head.
        degree_distance_dim: The dimensions to use while encoding the degree and distance.
        max_degree: The maximum degree of a node in the underlying graph.
        max_distance: The maximum distance of a node in the underlying graph.
        use_degree_distance: If true, encodes the degree and distance embeddings in the key/value computations.
        p_dropout: Probability with which the sum to the residual should be zeroed.
    """
    def __init__(self, n_heads, d_x, d_att, degree_distance_dim, max_degree, max_distance, use_degree_distance,
                 p_dropout):
        super(NeighborAttention, self).__init__()
        self.n_heads = n_heads
        self.d_x = d_x
        self.d_att = d_att
        self.temperature = d_att ** 0.5
        self.dropout_tensor = torch.nn.Parameter(torch.Tensor([1 - p_dropout, p_dropout]), requires_grad=False)

        # Degree and distance encodings.
        self.use_degree_distance = use_degree_distance
        if use_degree_distance:
            self.degree_embeddings_k = DegreeEmbedding(max_degree, degree_distance_dim)
            self.degree_embeddings_v = DegreeEmbedding(max_degree, degree_distance_dim)
            self.distance_embeddings_k = DistanceEmbedding(max_distance, degree_distance_dim)
            self.distance_embeddings_v = DistanceEmbedding(max_distance, degree_distance_dim)

        # Attention transform matrices.
        self.W_q = Linear(d_x, d_att * n_heads, bias=False)
        self.W_v = Linear(d_x + 2 * degree_distance_dim if use_degree_distance else d_x, d_att * n_heads, bias=False)
        self.W_k = Linear(d_x + 2 * degree_distance_dim if use_degree_distance else d_x, d_att * n_heads, bias=False)
        self.W_o = Linear(d_att * n_heads, d_x, bias=False)

        # Layer-norm
        self.layer_norm = LayerNorm(d_x)

        # Dropout.
        self.dropout = Dropout(p_dropout)

    def get_neighbor_attentions(self, xi, xj, degrees_j, distances_j, mask):
        """
        Get the attention values for each neighbor in xj[i] corresponding to each value of x[i].
        :param xi: (batch, d_x) = Input node attributes.
        :param xj: (batch, num_neighbors, d_x) = Node attributes for each neighbor of the input node.
        :param degrees_j: (batch, num_neighbors) = Degrees of the neighbor nodes.
        :param distances_j: (batch, num_neighbors) = Distances from the source node xi to each of the neighbor nodes.
        :param mask: (batch, num_neighbors) = Boolean values with True where the input in xj is valid.
        :return: query matrix q, key matrix k, value matrix v and the attention scores matrix alpha_ij
        """
        batch_size = xi.size(0)
        num_neighbors = max(1, xj.size(1))
        if mask.shape[1] == 0:
            # If all nodes in this batch have no neighbors, assume one neighbor.
            mask = torch.zeros(batch_size, 1, device=mask.device).bool()

        # Compute queries, keys and values.
        q = self.W_q(self.dropout(xi))[:, None, :]  # (batch, 1, d_att * n_heads)

        # shape({k,v}_xj) = (batch, num_neighbors, d_x + 2 * d_att) if degree/distance embeddings are encoded.
        #                 = (batch, num_neighbors, d_x)             otherwise.
        k_xj = xj
        v_xj = xj
        if self.use_degree_distance:
            degree_embed_k = self.degree_embeddings_k(degrees_j)
            degree_embed_v = self.degree_embeddings_v(degrees_j)
            distance_embed_k = self.distance_embeddings_k(distances_j)
            distance_embed_v = self.distance_embeddings_v(distances_j)
            k_xj = torch.cat([k_xj, degree_embed_k, distance_embed_k], dim=2)
            v_xj = torch.cat([v_xj, degree_embed_v, distance_embed_v], dim=2)

        # shape(k) = (batch, num_neighbors, d_att * n_heads)
        k = self.W_k(self.dropout(k_xj))
        # shape(v) = (batch, num_neighbors, d_att * n_heads)
        v = self.W_v(self.dropout(v_xj))

        # Separate out attention heads.
        q = q.view(batch_size, 1, self.n_heads, self.d_att)  # (batch, 1, n_heads, d_att)
        k = k.view(batch_size, num_neighbors, self.n_heads, self.d_att)  # (batch, num_neighbors, n_heads, d_att)
        v = v.view(batch_size, num_neighbors, self.n_heads, self.d_att)  # (batch, num_neighbors, n_heads, d_att)
        q = q.transpose(1, 2)  # (batch, n_heads, 1, d_att)
        k = k.transpose(1, 2)  # (batch, n_heads, num_neighbors, d_att)
        v = v.transpose(1, 2)  # (batch, n_heads, num_neighbors, d_att)

        # Compute s_ij for each neighbor.
        s_ij = torch.matmul(q / self.temperature, k.transpose(2, 3))  # (batch, n_heads, 1, num_neighbors)
        s_ij = s_ij.squeeze(2)  # (batch, n_heads, num_neighbors)
        if self.training:
            # Zero-out some neighbors so that the model attends to a stochastic neighborhood while training to increase
            # model robustness. The full neighborhood is available during evaluation.
            neighbor_dropout_mask = self.dropout_tensor.repeat(batch_size, 1)
            neighbor_dropout_mask = torch.multinomial(neighbor_dropout_mask, num_neighbors, replacement=True)
            mask = mask.masked_fill(neighbor_dropout_mask.bool(), False)
        # A node should always be able to attend to its own features.
        mask[:, 0] = True
        # Fill in -inf values in invalid sequence locations, so the Softmax value would be zero for them.
        # Mask has shape (batch, num_neighbors). We need to convert it to (batch, 1, num_neighbors) so we can broadcast
        # it to each of the attention heads.
        s_ij = s_ij.masked_fill(~mask[:, None, :], -1e7)
        alpha_ij = softmax(s_ij, dim=2)  # (batch, n_heads, num_neighbors)

        return q, k, v, alpha_ij

    def forward(self, xi, xj, degrees_j, distances_j, mask):
        """
        :param xi: (batch, d_x) = Input node attributes.
        :param xj: (batch, num_neighbors, d_x) = Node attributes for each neighbor of the input node.
        :param degrees_j: (batch, num_neighbors) = Degrees of the neighbor nodes.
        :param distances_j: (batch, num_neighbors) = Distances from the source node xi to each of the neighbor nodes.
        :param mask: (batch, num_neighbors) = Boolean values with True where the input in xj is valid.
        :returns attn: (batch, d_x) = Attended encoding for each of the input nodes in the batch.
                 alpha_ij: (batch, n_heads, num_neighbors) = Attention weights for each of the heads for each of the
                                                             neighbors corresponding to nodes in the batch.
        """
        batch_size = xi.size(0)
        _, _, v, alpha_ij = self.get_neighbor_attentions(xi, xj, degrees_j, distances_j, mask)

        # Compute neighbor-attended vector for each head.
        attn = torch.matmul(alpha_ij.unsqueeze(2), v).squeeze(2)  # (batch, n_heads, d_att)

        # Concatenate multiple heads and compute final attended output.
        attn = attn.view(batch_size, self.n_heads * self.d_att)  # (batch, n_heads * d_att)
        attn = self.W_o(self.dropout(attn))  # (batch, d_x)

        # Add residual connection and layer-normalize.
        attn = self.layer_norm(xi + self.dropout(attn))  # (batch, d_x)

        return attn, alpha_ij


class TwoLayerNetwork(Module):
    """
    A feed-forward neural network with two layers and ReLU activation and a residual connection.

    Args:
        d_x: Input and output dimensions.
        h_dim: Dimensions of the internal layer.
        p_dropout: Probability with which the sum to the residual should be zeroed.
    """
    def __init__(self, d_x, h_dim, p_dropout):
        super(TwoLayerNetwork, self).__init__()
        self.W1 = Linear(d_x, h_dim)
        self.W2 = Linear(h_dim, d_x)
        self.layer_norm = LayerNorm(d_x)
        self.dropout = Dropout(p_dropout)

    def forward(self, x):
        z = relu(self.W1(self.dropout(x)))
        z = self.W2(self.dropout(z))
        return self.layer_norm(x + self.dropout(z))


class NeighborEncoder(Module):
    """
    A neighbor encoder module for a node computes the encoded representation for a node as the sum of:
      (1) The neighbor-attended encoded representation of a node's neighborhood and its own attributes, and
      (2) The node's own original attributes
    followed by projection through a two-layer feed-forward neural network.

    Args:
        n_heads: The number of heads to use while attending to the neighbors.
        d_x: The input and output dimensions of the encoding.
        d_att: The dimensions to use per attention head while computing attention queries, keys and values.
        degree_distance_dim: The dimensions to use while encoding the degree and distance.
        max_degree: The maximum degree of a node in the underlying graph.
        max_distance: The maximum distance of a node in the underlying graph.
        fc_dim: The hidden layer dimensions to use in the two layer fully connected network.
        use_degree_distance: Encode the degrees and distances of neighbors in the attention mechanism.
    """
    def __init__(self, n_heads, d_x, d_att, degree_distance_dim, max_degree, max_distance, fc_dim,
                 use_degree_distance=False, p_dropout=0.5):
        super(NeighborEncoder, self).__init__()

        # Neighbor-Attention module.
        self.neighbor_attention = NeighborAttention(n_heads, d_x, d_att, degree_distance_dim, max_degree, max_distance,
                                                    use_degree_distance, p_dropout=p_dropout)

        # Two-layer fully-connected network.
        self.fcn = TwoLayerNetwork(d_x, fc_dim, p_dropout=p_dropout)

    def forward(self, xi, xj, degrees_j, distances_j, mask):
        """
        :param xi: (batch, d_x) = Input node attributes.
        :param xj: (batch, num_neighbors, d_x) = Node attributes for each neighbor of the input node.
        :param degrees_j: (batch, num_neighbors) = Degrees of the neighbor nodes.
        :param distances_j: (batch, num_neighbors) = Distances from the source node xi to each of the neighbor nodes.
        :param mask: (batch, num_neighbors) = A boolean mask with True in locations where xj is valid.
        :returns zi: (batch, d_x) = Attended encoding for each of the input nodes in the batch.
                 attn: (batch, n_heads, num_neighbors) = Attention weights for each of the heads for each of the
                                                         neighbors corresponding to nodes in the batch.
        """
        zi, attn = self.neighbor_attention(xi, xj, degrees_j, distances_j, mask)
        zi = self.fcn(zi)

        return zi, attn


class NodeEncoder(Module):
    """
    A node encoder module consists of multiple stacked neighbor encoders. It also adds degree and node-to-node distance
    embeddings to the input of the very first neighbor encoder in the stack.

    Args:
        n_encoders: The number of neighbor encoders to be stacked.
        max_degree: The maximum degree of any node in the graph.
        max_distance: The maximum distance between two nodes in the graph that needs to be encoded.
        n_heads: The number of heads to use while attending to the neighbors.
        d_x: The input dimensions of the encoding.
        d_out: The output dimensions of the encoding.
        d_att: The dimensions to use per attention head while computing attention queries, keys and values.
        degree_distance_dim: The dimensions to use while embedding degrees and distances.
        fc_dim: The hidden layer dimensions to use in the two layer fully connected network.
        p_dropout: The probability with which values should be dropped.
    """
    def __init__(self, n_encoders, max_degree, max_distance, n_heads, d_x, d_out, d_att, degree_distance_dim, fc_dim,
                 p_dropout):
        super(NodeEncoder, self).__init__()
        assert n_encoders > 0
        self.n_encoders = n_encoders
        self.max_distance = max_distance

        # To project `d_x` input features to `d_out`, so all the internal components work with `d_out` dimensions.
        self.input_projection = Linear(d_x, d_out, bias=False)
        self.dropout = Dropout(p_dropout)

        # Create and store each encoder layer.
        for encoder_no in range(n_encoders):
            # Only encode degrees and distances in the first layer.
            encoder = NeighborEncoder(n_heads, d_out, d_att, degree_distance_dim, max_degree, max_distance, fc_dim,
                                      use_degree_distance=encoder_no == 0, p_dropout=p_dropout)
            setattr(self, 'encoder_{i}'.format(i=encoder_no), encoder)

    def get_neighbor_attentions(self, graph_dataset, start_nodes, max_neighbors, attention_radius, mode):
        """
        Input parameter descriptions are the same as `forward` method below.
        :returns Attention matrix alpha_ij with shape (batch_size, n_heads, num_neighbors), with each of the
                 `num_neighbors` values being a probability score corresponding to the attention weight for that
                 neighbor. Returns the attention weights from the last encoder layer.
        """
        was_training = self.training
        self.eval()
        _, neighbors, alpha_ij = self.forward(graph_dataset, start_nodes, max_neighbors, attention_radius, mode)
        if was_training:
            self.train()
        return neighbors, alpha_ij  # (batch_size, n_heads, num_neighbors)

    def _forward_layer(self, xi, xj, degrees_j, distances_j, layer_no, mask):
        """
        :param xi: (batch, d_x) = Input node attributes.
        :param xj: (batch, num_neighbors, d_x) = Node attributes for each neighbor of the input node.
        :param degrees_j: (batch, num_neighbors) = Degrees of the neighbor nodes.
        :param distances_j: (batch, num_neighbors) = Distances from the source node xi to each of the neighbor nodes.
        :param layer_no: int = The layer number of the encoder to use. When stacking multiple encoders, we need to
                               specify which layer this is, as calls to deeper layers will require the xis and xjs
                               to be modified accordingly in the input.
        :param mask: (batch, num_neighbors) = A mask of Boolean values with True values when the input is valid.
                                              This is to account for the difference in the number of neighbors for
                                              different examples in the batch.
        :returns encoder_out: (batch, d_out) = Attended encoding for each of the input nodes in the batch.
                 attn: (batch, n_heads, num_neighbors) = Attention weights for each of the heads for each of the
                                                         neighbors corresponding to nodes in the batch.
        """
        # Compute the new encoded representation for this layer.
        if layer_no == 0:
            xi = relu(self.input_projection(self.dropout(xi)))
            xj = relu(self.input_projection(self.dropout(xj)))
        encoder = getattr(self, 'encoder_{i}'.format(i=layer_no))
        encoder_out, attn = encoder(xi, xj, degrees_j, distances_j, mask)

        return encoder_out, attn

    def _prepare_inputs(self, graph_dataset, nodes, neighbors, neighbors_to_distances, x, layer_no, mode):
        """
        Prepare inputs to the NodeEncoder in the format expected by `_forward_layer`.
        :param graph_dataset: The underlying graph as a pytorch_geometric.data.Data object.
        :param nodes: A List[int] of the starting nodes to compute encoder representations for.
        :param neighbors: List[List[int]]] of neighbors for each node in nodes.
        :param neighbors_to_distances: List[Dict[int -> int]] Neighbor to distance mapping for each node & neighbor.
        :param x: Dict[int -> Tensor] The model features for the nodes in `nodes` and their neighbors.
        :param layer_no: The layer number of the encoder.
        :param mode: One of `train`, `val` or `test` specifying the view of the graph to use.
        :return: Inputs expected by `_forward_layer`.
        """
        device = next(self.parameters()).device
        input_dims = next(iter(x.values())).shape[0]

        xi = torch.stack(list(np.vectorize(x.get, otypes=[torch.Tensor])(nodes)))  # (nodes, dx)
        xj = [torch.stack(list(np.vectorize(x.get, otypes=[torch.Tensor])(neighbors_i)))
              if neighbors_i else torch.zeros(1, input_dims, device=device)
              for neighbors_i in neighbors]  # (nodes, num_neighbors, dx)
        in_degrees = graph_dataset.train_in_degrees
        if mode == 'val':
            in_degrees = graph_dataset.val_in_degrees
        if mode == 'test':
            in_degrees = graph_dataset.test_in_degrees
        degrees_j = [graph_dataset.modified_degrees[node][neighbors_i] if node in graph_dataset.modified_degrees
                     else in_degrees[neighbors_i] if neighbors_i
                     else torch.zeros(1).long()
                     for node, neighbors_i in zip(nodes, neighbors)]  # (nodes, num_neighbors)
        distances_j = [np.vectorize(lambda n: neighbors_to_distances[batch_idx].get(n, self.max_distance))(neighbors_i)
                       if neighbors_i else np.ones(1, dtype=np.int64) * self.max_distance
                       for batch_idx, neighbors_i in enumerate(neighbors)]
        distances_j = [torch.from_numpy(distances_ij) for distances_ij in distances_j]  # (nodes, num_neighbors)

        # Mask for which inputs in the batch are valid.
        mask = [torch.ones(len(neighbors_i), device=device) for neighbors_i in neighbors]

        # Pad sequences with zeros, as different examples in the minibatch may have a different number of neighbors.
        xj = pad_sequence(xj, batch_first=True)
        degrees_j = pad_sequence(degrees_j, batch_first=True)
        distances_j = pad_sequence(distances_j, batch_first=True)
        mask = pad_sequence(mask, batch_first=True).bool()

        # Move to the appropriate device.
        degrees_j = degrees_j.to(device)
        distances_j = distances_j.to(device)

        return xi, xj, degrees_j, distances_j, layer_no, mask

    def forward(self, graph_dataset, start_nodes, max_neighbors, attention_radius, mode):
        """
        Performs a forward computation on for the NodeEncoder by making multiple passes to compute the embeddings at the
        nth neighbor encoder layer.
        Example of computation for a 3-layer stacked encoder with `attention_radius` = 1:
        1. Layer 1 embeddings should be computed for all nodes reachable within 2-hops of `start_nodes`.
        2. Layer 2 embedding should be computed for all nodes reachable within 1-hop of `start_nodes` using layer 1
           embeddings.
        3. Layer 3 embeddings should be computed only for `start_nodes` using Layer 2 embeddings of 1-hop nodes.
        The number of nodes in 1. is bounded by max_neighbors^num_encoder_layers.
        :param graph_dataset: The underlying graph as a pytorch_geometric.data.Data object.
        :param start_nodes: A List[int] of the starting nodes to compute final layer encoder representations for.
        :param max_neighbors: The maximum number of neighbors to use in the r-hop radius.
        :param attention_radius: The r-hop radius to attend to.
        :param mode: One of 'train', 'val' or 'test' which decides the nodes to be used.
        :returns embeds_layer: (batch, d_out) = Attended encoding for each of the input nodes in the batch.
                 neighbors: List[List[int]] = One item per node in the batch corresponding to the ids of the neighbors
                                              used in the computation.
                 attns_layer: (batch, n_heads, num_neighbors) = Attention weights for each of the heads for each of the
                                                                neighbors corresponding to nodes in the batch.
        """
        original_nodes = start_nodes
        layered_nodes = [set(start_nodes)]
        layer_neighbors = []
        for _ in range(1, self.n_encoders + 1):
            neighbors, neighbors_to_distances, _ = get_neighbors(graph_dataset, layered_nodes[-1], attention_radius,
                                                                 mode, max_neighbors)
            layer_neighbors.append((neighbors, neighbors_to_distances))
            layer_nodes = layered_nodes[-1].union([neighbor for neighbors_i in neighbors for neighbor in neighbors_i])
            layered_nodes.append(layer_nodes)
        # layered_nodes[i] represents the nodes to compute Layer i encoder embeddings for.
        layered_nodes = layered_nodes[::-1]
        layer_neighbors = layer_neighbors[::-1]

        # Perform multiple forward passes to compute Layer i encoder embeddings.
        x = {node: graph_dataset.x[node] for node in layered_nodes[0]}
        attns = {}
        neighbors_map = {}
        for layer_no, packed_data in enumerate(zip(layered_nodes[1:], layer_neighbors)):
            nodes, (neighbors, neighbors_to_distances) = packed_data
            inputs = self._prepare_inputs(graph_dataset, list(nodes), neighbors, neighbors_to_distances, x, layer_no,
                                          mode)
            embeds_layer, attns_layer = self._forward_layer(*inputs)
            x = {node: embed_node for node, embed_node in zip(nodes, embeds_layer)}
            attns = {node: attn_layer for node, attn_layer in zip(nodes, attns_layer)}
            neighbors_map = {node: neighbors_node for node, neighbors_node in zip(nodes, neighbors)}

        start_nodes_embeddings = torch.stack([x[node] for node in original_nodes])
        output_neighbors = [neighbors_map[node] for node in original_nodes]
        output_attns = torch.stack([attns[node] for node in original_nodes])
        return start_nodes_embeddings, output_neighbors, output_attns


class SupervisedNodeEncoder(Module):
    """
    A `NodeEncoder` model that predicts into `n_classes` classes, allowing for end-to-end training with a supervised
    objective.
    See all the corresponding arguments to `NodeEncoder` and `FCNetwork`.
    """
    def __init__(self, n_encoders, max_degree, max_distance, n_heads, d_x, d_out, d_att, degree_distance_dim, fc_dim,
                 n_classes, hidden_dims, activation, dropout):
        super(SupervisedNodeEncoder, self).__init__()
        self.node_encoder = NodeEncoder(n_encoders, max_degree, max_distance, n_heads, d_x, d_out, d_att,
                                        degree_distance_dim, fc_dim, dropout)
        self.fc_network = FCNetwork(d_out, n_classes, hidden_dims, activation, dropout=dropout)
        self.get_neighbor_attentions = self.node_encoder.get_neighbor_attentions

    def forward(self, graph_dataset, start_nodes, max_neighbors, attention_radius, mode):
        """
        See the corresponding parameters of `NodeEncoder`.
        """
        out_embeddings, neighbors, attn = self.node_encoder(
            graph_dataset, start_nodes, max_neighbors, attention_radius, mode)
        class_logits = self.fc_network(out_embeddings)  # (batch, n_classes)

        return class_logits, neighbors, attn


def margin_triplet_loss(margin, node_embeddings, positive_embeddings, negative_embeddings):
    """
    Computes a triplet margin loss using dot product to measure embedding similarity. The goal is to have node
    embeddings more similar to positive embeddings than negative embeddings by at least `margin`.
    :param margin: scalar float value of the margin to use as described above.
    :param node_embeddings: (batch, n_dim)
    :param positive_embeddings: (batch, n_dim)
    :param negative_embeddings: (batch, n_dim)
    :return: A single loss values averaged over the batch.
    """
    negative_dot = (node_embeddings * negative_embeddings).sum(dim=1)
    positive_dot = (node_embeddings * positive_embeddings).sum(dim=1)
    return relu(margin + negative_dot - positive_dot).mean()
