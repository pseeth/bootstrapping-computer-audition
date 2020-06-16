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
import glob
import tqdm
import warnings
import soundfile as sf

warnings.simplefilter("ignore")

@gin.configurable
class AudioSegmentDataset(nussl.datasets.MixSourceFolder):
    def __init__(self, folder, segment_length, hop_length, 
                 has_sources=False, **kwargs):
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.has_sources = has_sources
        super().__init__(folder, **kwargs)

    def get_items(self, folder):
        # get every audio file in the mix_folder.
        def _analyze_one_item(audio_file):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                audio_path = os.path.join(folder, self.mix_folder, audio_file)
                duration = sf.info(audio_path).duration
                starts = np.arange(0, duration, self.hop_length)
                starts[-1] = max(0, duration - self.segment_length)

                _items = [(audio_file, start) for start in starts]
            return _items
        
        mix_folder = os.path.join(folder, self.mix_folder)
        audio_files = []
        for ext in self.ext:
            _patterns = [f"{mix_folder}/**/*{ext}", f"{mix_folder}/**{ext}"]
            _audio_files = []
            for p in _patterns:
                _audio_files.extend(glob.glob(p))
            _audio_files = [
                os.path.relpath(
                    _audio_file, 
                    os.path.join(self.folder, self.mix_folder)
                ) for _audio_file in _audio_files
            ]
            audio_files.extend(_audio_files)

        items = []
        for audio_file in audio_files:
            _items = _analyze_one_item(audio_file)
            items.extend(_items)
        return items

    def process_item(self, item):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            audio_path, start = item
            mix_path = os.path.join(self.folder, self.mix_folder, audio_path)
            mix = self._load_audio_file(
                mix_path, offset=start, duration=self.segment_length
            )
            parent_dir = os.path.dirname(mix.path_to_input_file)
            new_file_name = f"{start} - {os.path.basename(mix.path_to_input_file)}"
            mix.path_to_input_file = os.path.join(parent_dir, new_file_name)

            output = {
                'mix': mix,
                'metadata': {
                    'labels': self.source_folders,
                    'start': start
                }
            }
            if self.has_sources:
                sources = {}
                for k in self.source_folders:
                    source_path = os.path.join(self.folder, k, audio_path)
                    if os.path.exists(source_path):
                        sources[k] = self._load_audio_file(
                            source_path, offset=start, duration=self.segment_length
                        )
                output['sources'] = sources
            return output

def _separate(separator, item, gpu_output):
    separator.audio_signal = item['mix']
    if gpu_output is not None:
        estimates = separator(gpu_output[0])
        separator.model_output = gpu_output[1]
    else:
        estimates = separator()
    return estimates

def _evaluate(evaluator, file_name, results_folder, 
              save_audio_path, extra_data=None):
    extra_data = {} if extra_data is None else extra_data
    scores = {}
    try:
        if evaluator is not None:
            scores.update(evaluator.evaluate())
    except:
        pass
    for key in extra_data:
        if key in scores:
            scores[key].update(extra_data[key])
        else:
            scores[key] = extra_data[key]
    output_path = os.path.join(
        results_folder, f"{file_name}.json")
    with open(output_path, 'w') as f:
        json.dump(scores, f, indent=2)
    os.makedirs(save_audio_path, exist_ok=True)
    return f"Done with {file_name}"

def separate_and_evaluate(separator, i, gpu_output, evaluator, 
                          results_folder, save_audio_path):
    with gin.config_scope('segment_and_separate'):
        test_dataset = build_dataset()
    item = test_dataset[i]
    file_name = item['mix'].file_name
    output_path = os.path.join(
        results_folder, f"{file_name}.json")
    if os.path.exists(output_path):
        return f"{file_name} exists!"
    if item['mix'].loudness() < -40:
        return f"{file_name} too quiet! Skipping."

    estimates = _separate(separator, item, gpu_output)
    extra_data = {}
    if hasattr(separator, 'confidence'):
        source_labels = evaluator.source_labels
        _confidence = {}
        confidence_approaches = [
            k for k in dir(nussl.ml.confidence) 
            if 'confidence' in k
        ]
        confidence_approaches = ['silhouette_confidence', 'posterior_confidence']
        for k in confidence_approaches:
            _confidence[k] = [float(separator.confidence(k, threshold=99))]
        for k in source_labels:
            extra_data[k] = _confidence

    if 'sources' in item:
        source_names = sorted(list(item['sources'].keys()))
        sources = [item['sources'][k] for k in source_names]
        evaluator.estimated_sources_list = estimates
        evaluator.true_sources_list = sources
    else:
        evaluator = None

    extra_data['metadata'] = {
        'original_path': item['mix'].path_to_input_file,
        'separated_path': [
            os.path.join(save_audio_path, f's{i}', file_name)
            for i in range(len(estimates))
        ]
    }

    _evaluate(evaluator, file_name, results_folder, 
              save_audio_path, extra_data)
    for i, e in enumerate(estimates):
        audio_dir = os.path.join(save_audio_path, f's{i}')
        audio_path = os.path.join(audio_dir, f'{file_name}')
        os.makedirs(audio_dir, exist_ok=True)
        e.write_audio_to_file(audio_path)
    
    return f"Done with {file_name}"

@gin.configurable
def segment_and_separate(output_folder, separation_algorithm, eval_class, 
                         block_on_gpu, num_workers, seed, save_audio_path, 
                         use_threadpool=False, num_sources=None):
    nussl.utils.seed(seed)
    logging.info(gin.operative_config_str())
    
    with gin.config_scope('segment_and_separate'):
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
            model_output = {
                k: v.cpu() for k, v in gpu_algorithm.model_output.items()
            }
            return gpu_output, model_output
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

            file_name = item['mix'].file_name
            output_path = os.path.join(
                results_folder, f"{file_name}.json")
            if os.path.exists(output_path):
                pbar.set_description(f"{file_name} exists!")
                pbar.update()
                continue

            pbar.set_description(f"Starting {item['mix'].file_name}")
            gpu_output = forward_on_gpu(item['mix'])
            kwargs = {'model_path': None} if set_model_to_none else {}

            empty_signal = nussl.AudioSignal(
                audio_data_array=np.random.rand(1, 100),
                sample_rate=100
            )
            separator = separation_algorithm(empty_signal, **kwargs)
            if set_model_to_none:
                separator.model = None
            if 'sources' in item:
                num_sources = len(item['sources'])

            dummy_signal_list = [
                nussl.AudioSignal(
                    audio_data_array=np.random.rand(1, 100),
                    sample_rate=100
                ) for _ in range(num_sources)
            ]
            evaluator = eval_class(dummy_signal_list, dummy_signal_list)
            args = (separator, i, gpu_output, evaluator, results_folder, 
                    save_audio_path)

            if num_workers == 1:
                desc = separate_and_evaluate(*args)
                pbar.update()
                pbar.set_description(desc)
            else:
                future = pool.submit(separate_and_evaluate, *args)
                future.add_done_callback(update)
