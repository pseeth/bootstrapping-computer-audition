import nussl
import glob
import os
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor as PoolExecutor
import gin
import tqdm
import logging

def process_one_item(dataset, i, output_directory, folder_name, resample_rate):
    name = dataset.musdb[i].name
    item = dataset[i]

    for key, signal in item['sources'].items():
        output_path = os.path.join(output_directory, folder_name, key)
        os.makedirs(output_path, exist_ok=True)
        if resample_rate is not None:
            signal.resample(resample_rate)
        signal.write_audio_to_file(
            os.path.join(output_path, f'{name}.wav'))
    
    mixture = sum(item['sources'].values())
    output_path = os.path.join(output_directory, folder_name, 'mixture')
    os.makedirs(output_path, exist_ok=True)
    mixture.write_audio_to_file(
        os.path.join(output_path, f'{name}.wav'))

    accompaniment = (
        item['sources']['drums'] + 
        item['sources']['bass'] + 
        item['sources']['other']
    )
    output_path = os.path.join(output_directory, folder_name, 'accompaniment')
    os.makedirs(output_path, exist_ok=True)
    accompaniment.write_audio_to_file(
        os.path.join(output_path, f'{name}.wav'))

    return f"Processed {name}"

@gin.configurable()
def process(input_directory, output_directory, resample_rate, num_workers):
    subsets = ['train', 'test']

    dataset_args = [
        {'subsets': ['train'], 'split': 'train', 'folder_name': 'train'},
        {'subsets': ['train'], 'split': 'valid', 'folder_name': 'valid'},
        {'subsets': ['test'], 'folder_name': 'test'},
    ]

    for dataset_arg in dataset_args:
        folder_name = dataset_arg.pop('folder_name')
        dataset = nussl.datasets.MUSDB18(input_directory, **dataset_arg)
        logging.info(f"Processing {folder_name}")
        pbar = tqdm.tqdm(total=len(dataset))

        def update(future):
            pbar.update()
            pbar.set_description(future.result())

        with PoolExecutor(num_workers) as ex:
            for i in range(len(dataset)):
                future = ex.submit(
                    process_one_item, dataset, i, output_directory, 
                    folder_name, resample_rate
                )
                future.add_done_callback(update)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str,
        help="""Path to gin configuration."""
    )
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    process()
