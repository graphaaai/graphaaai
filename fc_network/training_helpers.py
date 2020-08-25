import logging

from tqdm import tqdm
from torch.nn.functional import binary_cross_entropy_with_logits, cross_entropy
import torch

from eval_helpers import CROSS_ENTROPY

LOG = logging.getLogger(__name__)


def training_epoch(x, y, train_mask, model, optimizer, epoch, args):
    """
    Run one epoch of training on the dataset of inputs `x` and targets `y`, given the `model` and `optimizer`.
    :param x: A matrix of input arguments where dimension 1 has the features per example.
    :param y: The vector of targets per example.
    :param train_mask: A boolean mask with True specifying the rows that should be used for training.
    :param model: An instance of FCNetwork.
    :param optimizer: A torch optimizer to use.
    :param epoch: The current epoch for displaying statistics.
    :param args: The arguments passed to the program.
    :return: The running training loss value at the end of the epoch.
    """
    model.train()
    train_loss = torch.Tensor([0.])

    # Split input data into batches.
    examples = torch.arange(x.shape[0])[train_mask]
    examples = examples[torch.randperm(examples.shape[0])]
    total_steps = (train_mask.sum().item() + args.batch_size - 1) // args.batch_size
    minibatches = torch.split(examples, args.batch_size)
    if args.progress:
        minibatches = tqdm(minibatches, desc=f'Epoch {epoch + 1} [Train]', total=total_steps, leave=False)

    # Run training loop.
    loss_fn = cross_entropy if args.loss_fn == CROSS_ENTROPY else binary_cross_entropy_with_logits
    for minibatch_idxs in minibatches:
        # Zero out any previous gradients.
        optimizer.zero_grad()

        x_batch = x[minibatch_idxs]
        y_batch = y[minibatch_idxs]
        y_logits = model(x_batch)  # (batch, n_classes)

        loss = loss_fn(y_logits, y_batch)
        loss.backward()
        optimizer.step()

        train_loss += loss

    return train_loss.cpu().item() / total_steps


def validation_epoch(x, y, val_mask, model, epoch, args):
    """
    Run one epoch of validation on the dataset of inputs `x` and targets `y`, given the `model` and `optimizer`.
    :param x: A matrix of input arguments where dimension 1 has the features per example.
    :param y: The vector of targets per example.
    :param val_mask: A boolean mask with True specifying the rows that should be used for validation.
    :param model: An instance of FCNetwork.
    :param epoch: The current epoch for displaying statistics.
    :param args: The arguments passed to the program.
    :return: The validation loss value for the epoch.
    """
    model.eval()
    val_loss = torch.Tensor([0.])

    examples = torch.arange(x.shape[0])[val_mask]
    total_steps = (val_mask.sum().item() + args.batch_size - 1) // args.batch_size
    minibatches = torch.split(examples, args.batch_size)
    if args.progress:
        minibatches = tqdm(minibatches, desc=f'Epoch {epoch + 1} [Val]', total=total_steps, leave=False)

    loss_fn = cross_entropy if args.loss_fn == CROSS_ENTROPY else binary_cross_entropy_with_logits
    with torch.no_grad():
        for minibatch_idxs in minibatches:
            x_batch = x[minibatch_idxs]
            y_batch = y[minibatch_idxs]
            y_logits = model(x_batch)  # (batch, n_classes)

            loss = loss_fn(y_logits, y_batch)
            val_loss += loss

    return val_loss.cpu().item() / total_steps


def predict(x, mask, model, args):
    """
    Predict over the given input values using the model.
    :param x: A matrix of input arguments where dimension 1 has the features per example.
    :param mask: A boolean mask with True specifying the rows that should be predicted.
    :param model: An instance of FCNetwork.
    :param args: The arguments passed to the program.
    :return: A tensor of predictions.
    """
    model.eval()

    examples = torch.arange(x.shape[0])[mask]
    total_steps = (mask.sum().item() + args.batch_size - 1) // args.batch_size
    minibatches = torch.split(examples, args.batch_size)
    if args.progress:
        minibatches = tqdm(minibatches, desc=f'Predicting', total=total_steps, leave=False)

    all_predictions = []
    with torch.no_grad():
        for minibatch_idxs in minibatches:
            x_batch = x[minibatch_idxs]
            y_preds = model(x_batch)
            all_predictions.append(y_preds)

    return torch.cat(all_predictions, dim=0)
