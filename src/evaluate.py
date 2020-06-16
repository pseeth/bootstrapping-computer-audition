import nussl
import os
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tqdm
import gin
from .helpers import build_dataset
import logging
import numpy as np
from random import shuffle

def _separate(separator, item, gpu_output):
    separator.audio_signal = item['mix']
    if gpu_output is not None:
        estimates = separator(gpu_output)
    else:
        estimates = separator()
    # clear stft data so it can be pickled across processes
    for e in estimates:
        e.stft_data = None
    return estimates

def _evaluate(evaluator, file_name, results_folder, debug):
    scores = evaluator.evaluate()
    output_path = os.path.join(
        results_folder, f"{file_name}.json")
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=2)
    if debug:
        estimate_folder = output_path.replace(
            'results', 'audio').replace('json', '')
        os.makedirs(estimate_folder, exist_ok=True)
        for i, e in enumerate(estimates):
            audio_path = os.path.join(estimate_folder, f's{i}.wav')
            e.write_audio_to_file(audio_path)
    return f"Done with {file_name}"

def separate_and_evaluate(separator, i, gpu_output, evaluator, 
                          results_folder, debug):
    with gin.config_scope('test'):
        test_dataset = build_dataset()
    item = test_dataset[i]
    file_name = item['mix'].file_name
    output_path = os.path.join(
        results_folder, f"{file_name}.json")
    if os.path.exists(output_path):
        return f"{file_name} exists!"

    estimates = _separate(separator, item, gpu_output)
    source_names = sorted(list(item['sources'].keys()))
    sources = [item['sources'][k] for k in source_names]
    evaluator.estimated_sources_list = estimates
    evaluator.true_sources_list = sources
    return _evaluate(evaluator, file_name, 
                     results_folder, debug)

@gin.configurable
def evaluate(output_folder, separation_algorithm, eval_class, 
             block_on_gpu, num_workers, seed, debug=False, 
             use_threadpool=False):
    nussl.utils.seed(seed)
    logging.info(gin.operative_config_str())
    
    with gin.config_scope('test'):
        test_dataset = build_dataset()
    
    results_folder = os.path.join(output_folder, 'results')
    os.makedirs(results_folder, exist_ok=True)
    set_model_to_none = False

    if block_on_gpu:
        # make an instance that'll be used on GPU
        # has an empty audio signal for now
        gpu_algorithm = separation_algorithm(
            nussl.AudioSignal(), device='cuda')
        set_model_to_none = True

    def forward_on_gpu(audio_signal):
        # set the audio signal of the object to this item's mix
        if block_on_gpu:
            gpu_algorithm.audio_signal = audio_signal
            if hasattr(gpu_algorithm, 'forward'):
                gpu_output = gpu_algorithm.forward()
            elif hasattr(gpu_algorithm, 'extract_features'):
                gpu_output = gpu_algorithm.extract_features()
            return gpu_output
        else:
            return None

    pbar = tqdm.tqdm(total=len(test_dataset))
    
    PoolExecutor = (
        ThreadPoolExecutor 
        if use_threadpool 
        else ProcessPoolExecutor
    )

    with PoolExecutor(max_workers=num_workers) as pool:
        def update(future):
            desc = future.result()
            pbar.update()
            pbar.set_description(desc)

        indices = list(range(len(test_dataset)))
        shuffle(indices)
        for i in indices:
            item = test_dataset[i]

            pbar.set_description(f"Starting {item['mix'].file_name}")
            gpu_output = forward_on_gpu(item['mix'])
            kwargs = {'model_path': None} if set_model_to_none else {}

            empty_signal = nussl.AudioSignal(
                audio_data_array=np.random.rand(1, 100),
                sample_rate=100
            )
            separator = separation_algorithm(empty_signal, **kwargs)
            
            dummy_signal_list = [
                nussl.AudioSignal(
                    audio_data_array=np.random.rand(1, 100),
                    sample_rate=100
                ) for _ in range(len(item['sources']))
            ]
            evaluator = eval_class(dummy_signal_list, dummy_signal_list)
            args = (separator, i, gpu_output, evaluator, results_folder, debug)

            if num_workers == 1:
                desc = separate_and_evaluate(*args)
                pbar.update()
                pbar.set_description(desc)
            else:
                future = pool.submit(separate_and_evaluate, *args)
                future.add_done_callback(update)
