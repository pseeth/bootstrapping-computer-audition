import gin
from scaper import Scaper, generate_from_jams
import copy
import logging
import p_tqdm
import nussl
import os
import numpy as np

def _reset_event_spec(sc):
    sc.reset_fg_event_spec()
    sc.reset_bg_event_spec()

def check_mixture(path_to_mix):
    mix_signal = nussl.AudioSignal(path_to_mix)
    if mix_signal.rms() < .01:
        return False
    return True

def make_one_mixture(sc, path_to_file, num_sources, 
                     event_parameters, allow_repeated_label):
    """
    Creates a single mixture, incoherent. Instantiates according to
    the event parameters for each source.
    """
    check = False
    while not check:
        for j in range(num_sources):
            sc.add_event(**event_parameters)
        
        sc.generate(
            path_to_file, 
            path_to_file.replace('.wav', '.jams'), 
            no_audio=False,
            allow_repeated_label=allow_repeated_label,
            save_isolated_events=True,
        )
        _reset_event_spec(sc)
        check = check_mixture(path_to_file)

def instantiate_and_get_event_spec(sc, master_label, event_parameters):
    _reset_event_spec(sc)

    _event_parameters = copy.deepcopy(event_parameters)
    _event_parameters['label'] = ('const', master_label)
    sc.add_event(**_event_parameters)
    event = sc._instantiate_event(sc.fg_spec[-1])

    _reset_event_spec(sc)
    return sc, event

    

def make_one_mixture_coherent(sc, path_to_file, labels, event_parameters, 
                              allow_repeated_label):
    check = False
    while not check:
        sc, event = instantiate_and_get_event_spec(
            sc, labels[0], event_parameters)
        for label in labels:
            try:
                sc.add_event(
                    label=('const', label),
                    source_file=('const', event.source_file.replace(labels[0], label)),
                    source_time=('const', event.source_time),
                    event_time=('const', 0),
                    event_duration=('const', sc.duration),
                    snr=event_parameters['snr'],
                    pitch_shift=('const', event.pitch_shift),
                    time_stretch=('const', event.time_stretch)
                )
            except:
                logging.exception(
                    f"Got an error for {label} @ {_source_file}. Moving on...")
        sc.generate(
            path_to_file, 
            path_to_file.replace('.wav', '.jams'), 
            no_audio=False, 
            allow_repeated_label=allow_repeated_label,
            save_isolated_events=True,
        )
        sc.fg_spec = []
        check = check_mixture(path_to_file)

@gin.configurable
def make_scaper_datasets(scopes=['train', 'val']):
    for scope in scopes:
        with gin.config_scope(scope):
            mix_with_scaper()

@gin.configurable
def mix_with_scaper(num_mixtures, foreground_path, background_path, 
                    scene_duration, sample_rate, target_folder, 
                    event_parameters, num_sources=None, labels=None, 
                    coherent=False, allow_repeated_label=False, 
                    ref_db=-40, bitdepth=16, seed=0, num_workers=1):
    nussl.utils.seed(seed)
    os.makedirs(target_folder, exist_ok=True)

    scaper_seed = np.random.randint(100)
    logging.info('Starting mixing.')

    if num_sources is None and labels is None:
        raise ValueError("One of labels or num_sources must be set!")

    if coherent and labels is None:
        raise ValueError("Coherent mixing requires explicit labels!")

    generators = []

    if background_path is None:
        background_path = foreground_path

    for i in range(num_mixtures):
        sc = Scaper(
            scene_duration, 
            fg_path=foreground_path,
            bg_path=background_path,
            random_state=scaper_seed,
        )
        sc.ref_db = ref_db
        sc.sr = sample_rate
        sc.bitdepth = bitdepth
        generators.append(sc)
        scaper_seed += 1

    mix_func = make_one_mixture_coherent if coherent else make_one_mixture

    def arg_tuple(i):
        _args = (
            generators[i],
            os.path.join(target_folder, f'{i:08d}.wav'),
            labels if coherent else num_sources,
            event_parameters,
            allow_repeated_label
        )
        return _args

    args = [arg_tuple(i) for i in range(num_mixtures)]
    # do one by itself for testing
    mix_func(*args[0])

    args = list(zip(*args[1:]))
    args = [list(a) for a in args]

    # now do the rest in parallel
    p_tqdm.p_map(mix_func, *args, num_cpus=num_workers)
