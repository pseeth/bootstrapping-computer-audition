import gin
import logging
import os
from datetime import datetime
import sys
import nussl
import torch
import numpy as np
import warnings

@gin.configurable
def join_path(base_path, relative_path):
    return os.path.join(base_path, relative_path)

@gin.configurable
def output_folder(_output_folder=None):
    return _output_folder

@gin.configurable
def model_path(model_suffix):
    _output_folder = output_folder()
    _model_path = os.path.join(_output_folder, model_suffix)
    return _model_path

@gin.configurable
def remove_short_audio(min_length):
    def _remove_short_audio(self, item):
        processed_item = self.process_item(item)
        mix_length = processed_item['mix'].signal_duration
        if mix_length < min_length:
            return False
        return True
    return _remove_short_audio

@gin.configurable
def build_dataset(dataset_class, filter_func=None):
    if isinstance(dataset_class, type):
        # Not instantiated yet
        dataset = dataset_class()
    else:
        # Already instantiated
        dataset = dataset_class
    if filter_func is not None:
        dataset.filter_items_by_condition(filter_func)
    return dataset

@gin.configurable
def build_transforms(transform_names_and_args, cache_location):
    tfms = []
    for tfm_name, tfm_args in transform_names_and_args:
        if tfm_name == 'Cache':
            if cache_location is None:
                continue
            tfm_args['location'] = cache_location
        tfm = getattr(nussl.datasets.transforms, tfm_name)
        tfms.append(tfm(**tfm_args))
    return nussl.datasets.transforms.Compose(tfms)

@gin.configurable
def build_primitive_separator(kls):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        empty_signal = nussl.AudioSignal()
    return kls(empty_signal)

@gin.configurable
def build_primitive_clustering(scopes):
    separators = []
    for scope in scopes:
        with gin.config_scope(scope):
            separator = build_primitive_separator()
            separators.append(separator)
    return separators

@gin.configurable
def build_fixed_centers(weights):
    fixed_centers = np.array([
        [0 for i in range(sum(weights))],
        [1 for i in range(sum(weights))],
    ])
    return fixed_centers

def build_logger():
    _output_folder = output_folder()
    now = datetime.now()
    if _output_folder is not None:
        logging_file = os.path.join(
            _output_folder, 'logs', now.strftime("%Y.%m.%d-%H.%M.%S") + '.log')

        os.makedirs(os.path.join(_output_folder, 'logs'), exist_ok=True)
            
        logging.basicConfig(
            level = logging.INFO,
            format='%(asctime)s,%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
            handlers=[
                logging.FileHandler(logging_file),
                logging.StreamHandler(sys.stdout),
            ]
        )
    else:
        logging.basicConfig(
            level = logging.INFO,
            format='%(asctime)s,%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
        )

@gin.configurable
class DebugDataset(torch.utils.data.Dataset):
    """
    This dataset just wraps an existing dataset
    and always returns a random item. The length
    of the dataset is also set accordingly. This
    is to test whether a network can successfully
    overfit to a single item. That same item will 
    then be given for evaluation to make sure it gets
    good metrics.
    """
    def __init__(self, dataset, idx=None, dataset_length=20000, device='cuda'):
        if idx is None:
            idx = np.random.randint(len(dataset))
        
        self.dataset = dataset
        self.idx = idx
        self.dataset_length = dataset_length

        for attr in dir(dataset):
            if not attr.startswith('__'):
                setattr(self, attr, getattr(dataset, attr))

    def __getitem__(self, i):
        return self.dataset[self.idx]

    def __len__(self):
        return self.dataset_length
