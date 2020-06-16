import nussl
import torch
import numpy as np
import gin
import ignite
import copy
import pickle
import os
from .helpers import output_folder

@gin.configurable
def add_train_handlers(engine, model, scheduler, optimizer, 
                       train_closure, device, handler_names):
    for handler_name in handler_names:
        if handler_name == "add_clip_gradient_handler":
            add_clip_gradient_handler(engine, model)
        elif handler_name == "add_lr_scheduler_handler":
            add_lr_scheduler_handler(engine, scheduler)
        elif handler_name == "add_autoclip_gradient_handler":
            add_autoclip_gradient_handler(engine, model, train_closure)
        elif handler_name == "add_inspect_gradient":
            add_inspect_gradient(engine, model)
        elif handler_name == "add_auto_balance_loss":
            add_auto_balance_loss(engine, train_closure, device)
        elif handler_name == "early_stopping":
            add_early_stopping(engine)
        elif handler_name == "add_record_batch_and_loss_info_handler":
            add_record_batch_and_loss_info_handler(engine, train_closure)

@gin.configurable
def add_early_stopping(engine, patience=10):
    def score_function(engine):
        val_loss = engine.state.epoch_history['validation/loss'][-1]
        return -val_loss
    
    handler = ignite.handlers.EarlyStopping(
        patience=patience, score_function=score_function, trainer=engine
    )
    engine.add_event_handler(
        nussl.ml.train.ValidationEvents.VALIDATION_COMPLETED, 
        handler
    )

def _get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.requires_grad:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm 

@gin.configurable
def add_clip_gradient_handler(engine, model, clip_value):
    @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
    def clip_gradient(engine):
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

def add_lr_scheduler_handler(engine, scheduler):
    @engine.on(nussl.ml.train.ValidationEvents.VALIDATION_COMPLETED)
    def step_scheduler(engine):
        val_loss = engine.state.epoch_history['validation/loss'][-1]
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

@gin.configurable
def add_record_batch_and_loss_info_handler(engine, train_closure):
    loss_history_by_item = {}
    record_closure = copy.deepcopy(train_closure)
    keep_loss_tuples = []

    for loss_tuple in record_closure.losses:
        if hasattr(loss_tuple[0], 'reduction'):
            loss_tuple[0].reduction = 'none'
            keep_loss_tuples.append(loss_tuple)
        elif hasattr(loss_tuple[0], 'loss_function'):
            loss_tuple.loss_function.reduction = 'none'
            keep_loss_tuples.append(loss_tuple)
    
    record_closure.losses = keep_loss_tuples

    def append_loss_to_history(indices, loss_by_item):
        indices = indices.cpu().data.numpy().astype(int).tolist()
        for rel_idx, idx in enumerate(indices):
            if idx not in loss_history_by_item:
                loss_history_by_item[idx] = {}
            for key in loss_by_item:
                if key not in loss_history_by_item[idx]:
                    loss_history_by_item[idx][key] = []
                loss_history_by_item[idx][key].append(
                    loss_by_item[key][rel_idx].item()
                )

    @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
    def record_batch_and_loss_info(engine):
        indices = engine.state.batch['index']
        model_output = engine.state.model_output
        loss_output = record_closure.compute_loss(model_output, engine.state.batch)
        loss_by_item = {}
        for key in loss_output:
            shape = loss_output[key].shape
            if len(shape) > 0:
                loss_by_item[key] = loss_output[key].mean(
                    dim=tuple(range(1, len(shape))))
        append_loss_to_history(indices, loss_by_item)

    @engine.on(nussl.ml.train.ValidationEvents.VALIDATION_COMPLETED)
    def save_loss_history(engine):
        output_file = os.path.join(
            output_folder(), 'loss_history.pth'
        )
        with open(output_file, 'wb') as f:
            pickle.dump(loss_history_by_item, f)

@gin.configurable
def add_inspect_gradient(engine, model):
    @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
    def inspect_gradient(engine):
        obs_grad_norm = _get_grad_norm(model)
        if 'grad_norm' not in engine.state.iter_history:
            engine.state.iter_history['grad_norm'] = []
        engine.state.iter_history['grad_norm'].append(obs_grad_norm)

@gin.configurable
def add_auto_balance_loss(engine, train_closure, device, 
                          ref_percentile=100, update_frequency=1):
    n_losses = len(train_closure.losses)
    scale = 1 / n_losses
    loss_weights = torch.nn.ParameterList([
        torch.nn.Parameter(torch.ones(1).to(device))
        for _ in range(n_losses)
    ])

    weights_by_key = {}
    replaced_losses = []
    original_weight = {}
    
    # Replace weights with updatable parameter
    for weight, loss_tuple in zip(loss_weights, train_closure.losses):
        _loss_tuple = list(loss_tuple)
        original_weight[_loss_tuple[-1]] = loss_tuple[1]
        if loss_tuple[1] != 0:
            _loss_tuple[1] = weight
            replaced_losses.append(tuple(_loss_tuple))
            weights_by_key[_loss_tuple[-1]] = weight
    
    sorted_keys = sorted(list(weights_by_key.keys()))
    train_closure.losses = replaced_losses

    # Setting up for least squares problem
    off_diagonal = np.eye(n_losses) - 1
    diagonal = (n_losses - 1) * np.eye(n_losses)
    
    A = off_diagonal + diagonal
    B = np.zeros(1 + n_losses)
    B[-1] = 1

    ratios = np.array([
        original_weight[key] for key in sorted_keys
    ])
    W = 1 / ratios

    loss_history = {
        key: [] for key in sorted_keys
    }

    @engine.on(ignite.engine.Events.ITERATION_COMPLETED)
    def auto_balance_weights(engine):
        if engine.state.iteration % update_frequency == 0:
            L = []
            for key in sorted_keys:
                val = weights_by_key[key]
                loss_key = f'weight/{key}'
                if loss_key not in engine.state.iter_history:
                    engine.state.iter_history[loss_key] = []
                engine.state.iter_history[loss_key].append(val.item())

                loss_history[key].append(engine.state.output[key])
            
                L.append(
                    np.percentile(
                        loss_history[key],
                        ref_percentile)
                )
            
            L = np.array(L)
            _A = A * L * W
            _A = np.vstack([_A, np.ones(n_losses)])

            # Solve with least squares for weights so each
            # loss function matches what is given in ratios.
            X = np.linalg.lstsq(_A, B, rcond=None)[0]

            # Set the weights appropriately
            for i, key in enumerate(sorted_keys):
                weights_by_key[key].data[0] = X[i]

@gin.configurable
def add_autoclip_gradient_handler(engine, model, train_closure, clip_percentile):
    # Keep track of the history of gradients and select a cutoff
    # to clip values to based on percentile.
    grad_history = []

    @engine.on(nussl.ml.train.BackwardsEvents.BACKWARDS_COMPLETED)
    def autoclip_gradient(engine):
        # ignore some iterations as the grads are not useful yet
        obs_grad_norm = _get_grad_norm(model)
        grad_history.append(obs_grad_norm)

        clip_value = np.percentile(grad_history, clip_percentile)

        if 'grad_clip' not in engine.state.iter_history:
            engine.state.iter_history['grad_clip'] = []
        if 'grad_norm' not in engine.state.iter_history:
            engine.state.iter_history['grad_norm'] = []
        
        engine.state.iter_history['grad_clip'].append(clip_value)
        engine.state.iter_history['grad_norm'].append(obs_grad_norm)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)