import gin
import argparse
from src import (
    train, 
    cache, 
    analyze, 
    instantiate, 
    evaluate, 
    helpers, 
    make_scaper_datasets,
    segment_and_separate,
    construct_symlink_folder
)
import nussl
import subprocess
from src.helpers import build_logger, DebugDataset, output_folder
import os
import copy
import logging

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'func', type=str, 
        help= (
            "What function to run, given the configuration. "
            "Choices are train, evaluate, analyze, all, cache, instantiate, resume, "
            "debug. "
        )
    )
    parser.add_argument(
        "-c", "--config", nargs="+", type=str,
        help="List of .gin files containing bindings for relevant functions."
    )
    parser.add_argument(
        '-o', '--output_folder', type=str,
        help='If using instantiate command, need an output folder to put the compiled .gin config.'
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    special_commands = ['all', 'debug', 'resume']

    if args.func not in special_commands:
        if args.func not in globals():
            raise ValueError(f"No matching function named {args.func}!")
        func = globals()[args.func]

    for _config in args.config:
        if _config is not None:
            gin.parse_config_file(_config)

    build_logger()

    if args.func == 'debug':
        # overfit to a single batch for a given length.
        # save the model
        # evaluate it on that same sample
        # do this via binding parameters to gin config
        # then set args.func = 'all'
        debug_output_folder = os.path.join(
            helpers.output_folder(), 'debug')
        gin.bind_parameter(
            'output_folder._output_folder', 
            debug_output_folder
        )
        with gin.config_scope('train'):
            train_dataset = helpers.build_dataset()
        
        test_dataset = copy.deepcopy(train_dataset)
        test_dataset.transform = None
        test_dataset.cache_populated = False

        train_dataset = DebugDataset(train_dataset)
        val_dataset = copy.deepcopy(train_dataset)
        val_dataset.dataset_length = 1

        test_dataset = DebugDataset(test_dataset)
        test_dataset.dataset_length = 1
        test_dataset.idx = train_dataset.idx

        gin.bind_parameter('train/build_dataset.dataset_class', train_dataset)
        gin.bind_parameter('val/build_dataset.dataset_class', val_dataset)
        gin.bind_parameter('test/build_dataset.dataset_class', test_dataset)

        gin.bind_parameter('train.num_epochs', 1)
        gin.bind_parameter('evaluate.debug', True)

        args.func = 'all'

    if args.func == 'resume':
        gin.bind_parameter('train.resume', True)
        args.func = 'all'
    
    if args.func == 'all':
        if output_folder() is None:
            alert = "Run instantiate first!"
            border = ''.join(['=' for _ in alert])
            logging.exception(
                f'\n\n{border}\n'
                f'{alert}\n'
                f'{border}\n'
            )
        train()
        evaluate()
        analyze()
       
    elif args.func == 'cache':
        def _setup_for_cache(scope):
            with gin.config_scope(scope):
                _dataset = helpers.build_dataset()
                _dataset.cache_populated = False
                gin.bind_parameter(
                    f'{scope}/build_dataset.dataset_class', 
                    _dataset
                )
        for scope in ['train', 'val']:
            _setup_for_cache(scope)
        cache()
    elif args.func == 'instantiate':
        func(args.output_folder)
    else:
        func()

    if args.func == 'evaluate':
        analyze()
