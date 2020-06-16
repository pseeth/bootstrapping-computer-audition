import gin
import scaper
import nussl
import copy
import warnings
import numpy as np
import tqdm
import glob
import os
import hashlib
import pickle

def _reset_event_spec(sc):
    sc.reset_fg_event_spec()
    sc.reset_bg_event_spec()

def window_stack(a, stepsize=1, width=20):
    return np.vstack(
        a[i:1+i-width or None:stepsize] for i in range(0,width)
    )

def instantiate_and_get_event_spec(sc, master_label, event_parameters, 
                                   loud_regions=None):
    _reset_event_spec(sc)

    event_parameters['label'] = ('const', master_label)
    sc.add_event(**event_parameters)
    event = sc._instantiate_event(sc.fg_spec[-1])

    if loud_regions is not None:
        _event_parameters = event_parameters.copy()
        _event_parameters['source_file'] = ('const', event.source_file)
        loudness_data = loud_regions[event.source_file]
        loudness = loudness_data[1, :]
        loudness[~np.isfinite(loudness)] = -80
        loud_idx = np.min(window_stack(loudness), axis=0) > -80
        if loud_idx.sum() != 0:
            loud_times = loudness_data[0, :len(loud_idx)]
            loud_times = list(loud_times[loud_idx])
            _event_parameters['source_time'] = ('choose', loud_times)

            sc.add_event(**_event_parameters)
            event = sc._instantiate_event(sc.fg_spec[-1])

    _reset_event_spec(sc)
    return sc, event

def _convert_to_output(dataset, jam, soundscape_audio_data, 
                       source_audio_data, sample_rate):
    ann = jam.annotations.search(namespace='scaper')[0]
    soundscape_audio_data = np.nan_to_num(soundscape_audio_data)
    source_audio_data = [
        np.nan_to_num(x) for x in source_audio_data
    ]

    mix = dataset._load_audio_from_array(
        soundscape_audio_data, sample_rate)
    sources = {}

    for i, event_spec in enumerate(ann):
        label = event_spec.value['label']
        label_count = 0
        for k in sources:
            if label in k:
                label_count += 1
        label = f"{label}::{label_count}"
        sources[label] = dataset._load_audio_from_array(
            source_audio_data[i], sample_rate)

    output = {
        'mix': mix,
        'sources': sources,
        'metadata': {
            'scaper': jam,
            'labels': ann.sandbox.scaper['fg_labels'],
        }
    }
    return output


@gin.configurable
def create_mix_coherent_closure(fg_path, bg_path, scene_duration, 
                                sample_rate, fg_event_parameters, labels, 
                                num_channels=1, ref_db=-40, bitdepth=16, 
                                num_bg_sources=0, bg_event_parameters=None,
                                quick_pitch_time=False, ignore_warnings=True,
                                loud_regions=None):
    
    def mix_closure(dataset, i):
        sc = scaper.Scaper(
            scene_duration,
            fg_path=fg_path,
            bg_path=bg_path,
            random_state=i,
        )
        sc.ref_db = ref_db
        sc.sr = sample_rate
        sc.n_channels = num_channels
        sc.bitdepth = bitdepth
        sc.fade_in_len = 0
        sc.fade_out_len = 0

        if isinstance(fg_event_parameters, list):
            assert len(fg_event_parameters) == len(labels)
            event_parameters = copy.deepcopy(fg_event_parameters)
        else:
            event_parameters = [fg_event_parameters for _ in labels]
        
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter("ignore")

            sc, event = instantiate_and_get_event_spec(
                sc, labels[0], event_parameters[0], loud_regions
            )

            for i, label in enumerate(labels):
                _pitch_shift = ('const', event.pitch_shift)
                if event.pitch_shift is None:
                    _pitch_shift = None
                _time_stretch = ('const', event.time_stretch)
                if event.time_stretch is None:
                    _time_stretch = None
                try:
                    sc.add_event(
                        label=('const', label),
                        source_file=('const', event.source_file.replace(labels[0], label)),
                        source_time=('const', event.source_time),
                        event_time=('const', event.event_time),
                        event_duration=('const', event.event_duration),
                        snr=event_parameters[i]['snr'],
                        pitch_shift=_pitch_shift,
                        time_stretch=_time_stretch
                    )
                except:
                    sc.add_event(**event_parameters[i])
            for i in range(num_bg_sources):
                sc.add_background(**bg_event_parameters)
            jam, soundscape_audio_data, source_audio_data = sc.generate(
                None, None, quick_pitch_time=quick_pitch_time, 
                disable_instantiation_warnings=ignore_warnings,
            )
        return _convert_to_output(
            dataset, jam, soundscape_audio_data, 
            source_audio_data, sample_rate
        )
    return mix_closure

@gin.configurable
def get_loud_regions(fg_path, segment_size_in_seconds, 
                     hop_size_in_seconds, cache_location, 
                     overwrite=False):
    audio_files = glob.glob(f"{fg_path}/**/*.wav")

    hash_file = hashlib.sha224(" ".join(audio_files).encode('utf-8')).hexdigest()
    hash_file = os.path.join(cache_location, hash_file)
    print(f"Writing or looking for {hash_file}")

    if os.path.exists(hash_file) and not overwrite:
        with open(hash_file, 'rb') as f:
            loud_regions = pickle.load(f)
            return loud_regions

    pbar = tqdm.trange(len(audio_files))
    loud_regions = {}

    for a in audio_files:    
        audio_signal = nussl.AudioSignal(a)

        segment_size = segment_size_in_seconds * audio_signal.sample_rate
        hop_size = hop_size_in_seconds * audio_signal.sample_rate
        starts = list(np.arange(0, audio_signal.signal_length, hop_size))
        ends = list(np.arange(segment_size, audio_signal.signal_length, hop_size))
        ends.append(-1)
        loudness_over_time = []
        
        for start, end in zip(starts, ends):
            if end - start >= segment_size:
                audio_signal.set_active_region(start, end)
                loudness = audio_signal.loudness()
            else:
                loudness = -np.inf
            loudness_over_time.append(loudness)
        
        audio_signal.set_active_region_to_default()
        starts_in_seconds = np.linspace(
            0, audio_signal.signal_duration, len(loudness_over_time)
        )
        data = np.vstack([starts_in_seconds, loudness_over_time])
        path_to_file = os.path.abspath(a)
        loud_regions[path_to_file] = data
            
        pbar.update()
        pbar.set_description(f"Found loud regions for {audio_signal.file_name}")

    with open(hash_file, 'wb') as f:
        pickle.dump(loud_regions, f)
    return loud_regions

@gin.configurable
def create_mix_incoherent_closure(fg_path, bg_path, scene_duration, 
                                  sample_rate, fg_event_parameters, 
                                  num_sources, num_channels=1, ref_db=-40, 
                                  bitdepth=16, allow_repeated_label=False, 
                                  num_bg_sources=0, bg_event_parameters=None,
                                  quick_pitch_time=False, ignore_warnings=True, 
                                  loud_regions=None):
    def mix_closure(dataset, i):
        sc = scaper.Scaper(
            scene_duration,
            fg_path=fg_path,
            bg_path=bg_path,
            random_state=i,
        )
        sc.ref_db = ref_db
        sc.sr = sample_rate
        sc.bitdepth = bitdepth
        sc.n_channels = num_channels
        sc.fade_in_len = 0
        sc.fade_out_len = 0

        if isinstance(fg_event_parameters, list):
            assert len(fg_event_parameters) == len(labels)
            event_parameters = fg_event_parameters
        else:
            event_parameters = [fg_event_parameters for _ in range(num_sources)]
        with warnings.catch_warnings():
            if ignore_warnings:
                warnings.simplefilter("ignore")
            for i in range(num_sources):
                sc.add_event(**event_parameters[i])
            for i in range(num_bg_sources):
                sc.add_background(**bg_event_parameters)
            jam, soundscape_audio_data, source_audio_data = sc.generate(
                None, None, allow_repeated_label=allow_repeated_label, 
                quick_pitch_time=quick_pitch_time,
                disable_instantiation_warnings=ignore_warnings,
            )
        return _convert_to_output(
            dataset, jam, soundscape_audio_data, 
            source_audio_data, sample_rate
        )
    return mix_closure

@gin.configurable
def create_coherent_incoherent_closure(balance):
    coherent = create_mix_coherent_closure()
    incoherent = create_mix_incoherent_closure()

    def mix_closure(dataset, i):
        if np.random.rand() >= balance:
            return coherent(dataset, i)
        else:
            return incoherent(dataset, i)
    return mix_closure
