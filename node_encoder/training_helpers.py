import copy
import logging

from sklearn.metrics import accuracy_score, f1_score
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy, l1_loss
from tqdm import tqdm
import torch

from eval_helpers import CROSS_ENTROPY
from eval_helpers import raw_output_to_prediction
from node_encoder.minibatch_utils import get_mask, get_triplets, minibatch_generator, perturb_graph
from node_encoder.minibatch_utils import prediction_minibatch_generator
from node_encoder.minibatch_utils import unperturb_graph
from node_encoder.node_encoder import margin_triplet_loss

LOG = logging.getLogger(__name__)


class EarlyStopper(object):
    """
    Helper class to stop once the validation loss does not decrease for a fixed number of epochs and restore the best
    weights.
    """
    def __init__(self, early_stopping_epochs):
        self.early_stopping_epochs = early_stopping_epochs
        self.best_val_loss = 1e9
        self.epochs_since_decrease = 0
        self.best_weights = None

    def should_stop(self, model, val_loss):
        """
        Given the next validation loss, compares it to the previous validation loss and decides whether early stopping
        should be done. If so, it restores the best weights to the `model`. If not, it updates the best weight if the
        current validation loss is lower than the previous validation loss.
        :param model: The model file to load and store weights from.
        :param val_loss: The current epoch val loss.
        :return: bool = Whether training should stop.
        """
        if val_loss < self.best_val_loss:
            self.epochs_since_decrease = 0
            self.best_val_loss = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.epochs_since_decrease += 1
            if self.epochs_since_decrease >= self.early_stopping_epochs:
                model.load_state_dict(self.best_weights)
                return True

        return False


def training_epoch(graph_dataset, model, optimizer, epoch, args):
    """
    Run one epoch of training on the `graph_dataset` given the `model` and `optimizer`.
    :param graph_dataset: The graph represented as a pytorch_geometric.data.Data object.
    :param model: A `NodeEncoder` model instance.
    :param optimizer: A torch optimizer to use.
    :param epoch: The current epoch for displaying statistics.
    :param args: The arguments passed to the program.
    """
    model.train()
    mode = 'train'
    steps_per_graph = [(getattr(graph, f'{mode}_mask').sum().item() + args.batch_size - 1) // args.batch_size
                       for graph in graph_dataset.graphs]
    num_steps = sum(steps_per_graph)
    use_attention_negatives = epoch + 1 >= args.attention_negatives_start_epoch
    if use_attention_negatives:
        LOG.info('Using attention negatives.')
    minibatches = minibatch_generator(graph_dataset, args.batch_size, args.attention_radius,
                                      negative_radius=args.negative_radius, positive_radius=args.positive_radius,
                                      use_attention_negatives=use_attention_negatives, model=model, mode=mode,
                                      max_neighbors=args.max_neighbors)
    if args.progress:
        minibatches = tqdm(minibatches, total=num_steps, desc=f'Epoch {epoch+1} [Train]', leave=False)
    beta = 0
    if epoch + 1 >= args.perturbed_loss_start_epoch:
        beta = min(args.beta_end, args.beta_start + args.beta_increment * (epoch + 1 - args.perturbed_loss_start_epoch))
    train_loss = torch.Tensor([0.])
    nodes_trained_on = 0
    for nodes_minibatch, pos_minibatch, neg_minibatch in minibatches:
        # Zero out any previous gradients.
        optimizer.zero_grad()
        # Attention values are ignored for now.
        node_embeddings, _, _ = model(graph_dataset.active_graph, nodes_minibatch, args.max_neighbors,
                                      args.attention_radius, mode)
        pos_embeddings, _, _ = model(graph_dataset.active_graph, pos_minibatch, args.max_neighbors,
                                     args.attention_radius, mode)
        neg_embeddings, _, _ = model(graph_dataset.active_graph, neg_minibatch, args.max_neighbors,
                                     args.attention_radius, mode)
        loss = margin_triplet_loss(args.loss_margin, node_embeddings, pos_embeddings, neg_embeddings)

        # Track how many nodes we actually trained on (were able to generate triplets for).
        nodes_trained_on += len(set(nodes_minibatch))

        # If we need to use the perturbed loss, perturb the graph and generate the embeddings for the same.
        if beta > 0:
            perturb_graph(graph_dataset.active_graph, model, nodes_minibatch, args.max_neighbors, mode,
                          args.attention_radius, args.max_perturbations_per_node)
            nodes_minibatch, pos_minibatch, neg_minibatch = get_triplets(
                graph_dataset.active_graph, nodes_minibatch, args.attention_radius, args.positive_radius,
                args.negative_radius, mode, use_attention_negatives, model, args.max_neighbors)

            # If we were able to construct triplets from the perturbed graph, add the perturbed graph loss term.
            if nodes_minibatch:
                new_node_embeddings, _, _ = model(graph_dataset.active_graph, nodes_minibatch, args.max_neighbors,
                                                  args.attention_radius, mode)
                new_pos_embeddings, _, _ = model(graph_dataset.active_graph, pos_minibatch, args.max_neighbors,
                                                 args.attention_radius, mode)
                new_neg_embeddings, _, _ = model(graph_dataset.active_graph, neg_minibatch, args.max_neighbors,
                                                 args.attention_radius, mode)
                loss_perturbed = margin_triplet_loss(args.loss_margin, new_node_embeddings, new_pos_embeddings,
                                                     new_neg_embeddings)
                # Combine the regular and perturbed graph losses.
                loss = (1 - beta) * loss + beta * loss_perturbed

            unperturb_graph(graph_dataset.active_graph)

        loss.backward()
        optimizer.step()

        train_loss += loss

    return train_loss.cpu().item() / num_steps, nodes_trained_on


def validation_epoch(graph_dataset, model, epoch, args):
    """
    Run one epoch of validation on the `graph_dataset` given the `model`.
    :param graph_dataset: The graph represented as a pytorch_geometric.data.Data object.
    :param model: A `NodeEncoder` model instance.
    :param epoch: The current epoch for displaying statistics.
    :param args: The arguments passed to the program.
    """
    model.eval()
    mode = 'val'
    steps_per_graph = [(getattr(graph, f'{mode}_mask').sum().item() + args.batch_size - 1) // args.batch_size
                       for graph in graph_dataset.graphs]
    num_steps = sum(steps_per_graph)
    minibatches = minibatch_generator(graph_dataset, args.batch_size, args.attention_radius, args.positive_radius,
                                      args.negative_radius, model=model, is_val=True, use_attention_negatives=False,
                                      mode=mode)
    if args.progress:
        minibatches = tqdm(minibatches, desc=f'Epoch {epoch + 1} [Val]', total=num_steps, leave=False)

    val_loss = torch.Tensor([0.])
    nodes_validated_on = 0
    with torch.no_grad():
        for nodes_minibatch, positives_minibatch, negatives_minibatch in minibatches:
            # Attention values are ignored for now.
            node_embeddings, _, _ = model(graph_dataset.active_graph, nodes_minibatch, args.max_neighbors,
                                          args.attention_radius, mode)
            pos_embeddings, _, _ = model(graph_dataset.active_graph, positives_minibatch, args.max_neighbors,
                                         args.attention_radius, mode)
            neg_embeddings, _, _ = model(graph_dataset.active_graph, negatives_minibatch, args.max_neighbors,
                                         args.attention_radius, mode)
            loss = margin_triplet_loss(args.loss_margin, node_embeddings, pos_embeddings, neg_embeddings)
            val_loss += loss

            nodes_validated_on += len(set(nodes_minibatch))

    return val_loss.cpu().item() / num_steps, nodes_validated_on


def training_epoch_end_to_end(graph_dataset, model, optimizer, args, epoch):
    """
    Run one epoch of training on the `graph_dataset` given the `model` and `optimizer` in an end-to-end manner, i.e.
    with the supervised objective.
    :param graph_dataset: The graph represented as a pytorch_geometric.data.Data object.
    :param model: A `NodeEncoder` model instance.
    :param optimizer: A torch optimizer to use.
    :param args: The arguments passed to the program.
    :return: The training loss.
    """
    model.train()
    mode = 'train'
    steps_per_graph = [(getattr(graph, f'{mode}_mask').sum().item() + args.batch_size - 1) // args.batch_size
                       for graph in graph_dataset.graphs]
    num_steps = sum(steps_per_graph)
    minibatches = prediction_minibatch_generator(graph_dataset, args.batch_size, mode, shuffle=True, labels=True)
    if args.progress:
        minibatches = tqdm(minibatches, total=num_steps, desc=f'Epoch {epoch+1} [Train]', leave=False)
    beta = 0
    if epoch + 1 >= args.perturbed_loss_start_epoch:
        beta = min(args.beta_end, args.beta_start + args.beta_increment * (epoch + 1 - args.perturbed_loss_start_epoch))
    total_loss = torch.Tensor([0.])
    loss_fn = cross_entropy if args.loss_fn == CROSS_ENTROPY \
        else binary_cross_entropy_with_logits
    for y_true, minibatch in minibatches:
        optimizer.zero_grad()

        y_logits, _, _ = model(graph_dataset.active_graph, minibatch, args.max_neighbors, args.attention_radius, mode)
        loss = loss_fn(y_logits, y_true)

        if beta > 0:
            perturb_graph(graph_dataset.active_graph, model, minibatch, args.max_neighbors, mode, args.attention_radius,
                          max_perturbations_per_node=args.max_perturbations_per_node)
            y_logits, _, _ = model(graph_dataset.active_graph, minibatch, args.max_neighbors, args.attention_radius,
                                   mode)
            loss_perturbed = loss_fn(y_logits, y_true)
            loss = (1 - beta) * loss + beta * loss_perturbed
            unperturb_graph(graph_dataset.active_graph)

        loss.backward()
        optimizer.step()

        total_loss += loss

    return total_loss.cpu().item() / num_steps


def validation_epoch_end_to_end(graph_dataset, model, args):
    """
    Run one epoch of validation on the `graph_dataset` given the `model` and `optimizer` in an end-to-end manner, i.e.
    with the supervised objective.
    :param graph_dataset: The graph represented as a pytorch_geometric.data.Data object.
    :param model: A `NodeEncoder` model instance.
    :param args: The arguments passed to the program.
    :return: The validation loss.
    """
    model.eval()
    mode = 'val'
    steps_per_graph = [(getattr(graph, f'{mode}_mask').sum().item() + args.batch_size - 1) // args.batch_size
                       for graph in graph_dataset.graphs]
    num_steps = sum(steps_per_graph)
    minibatches = prediction_minibatch_generator(graph_dataset, args.batch_size, mode, labels=True)
    if args.progress:
        minibatches = tqdm(minibatches, total=num_steps, desc=f'[Val]', leave=False)
    val_loss = torch.Tensor([0.])
    loss_fn = cross_entropy if args.loss_fn == CROSS_ENTROPY \
        else binary_cross_entropy_with_logits
    predictions = []
    labels = []
    with torch.no_grad():
        for y_true, minibatch in minibatches:
            y_logits, _, _ = model(graph_dataset.active_graph, minibatch, args.max_neighbors, args.attention_radius,
                                   mode)
            predictions.append(y_logits)
            labels.append(y_true)
            loss = loss_fn(y_logits, y_true)
            val_loss += loss

    predictions = torch.cat(predictions, dim=0)
    predictions = raw_output_to_prediction(predictions, args.loss_fn)
    labels = torch.cat(labels, dim=0)
    accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='micro')
    return val_loss.cpu().item() / num_steps, accuracy, f1


def predict(graph_dataset, model, args, mode):
    """
    Use the NodeEncoder `model` to predict on the nodes in the `graph_dataset` specified by the `mask` and return the
    embeddings or the class logits, depending on the model.
    :param graph_dataset: The graph represented as a pytorch_geometric.data.Data object.
    :param model: A `NodeEncoder` model instance.
    :param args: The arguments passed to the program.
    :param mode: One of 'train', 'val' or 'test' which decides which nodes are visible during graph traversal.
    :return: The embeddings for the nodes or the class probabilities for the nodes, depending on the model.
    """
    model.eval()
    steps_per_graph = [
        (getattr(graph, f'{mode}_mask').sum().item() + args.batch_size - 1) // args.batch_size
        for graph in graph_dataset.graphs
    ]
    num_steps = sum(steps_per_graph)
    minibatches = prediction_minibatch_generator(graph_dataset, args.batch_size, mode, labels=True)
    if args.progress:
        minibatches = tqdm(minibatches, total=num_steps, desc=f'Predicting', leave=False)

    predictions = []
    with torch.no_grad():
        for y_true, minibatch in minibatches:
            y_pred, _, _ = model(graph_dataset.active_graph, minibatch, args.max_neighbors, args.attention_radius,
                                 mode=mode)
            predictions.append(y_pred)

    return torch.cat(predictions, dim=0)
