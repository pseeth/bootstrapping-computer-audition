import glob
import json
import nussl
import os
import pandas as pd
import shutil
import numpy as np
import tqdm
import gin
import re
from tqdm import tqdm
import hashlib
import pickle
tqdm.pandas()

@gin.configurable
def construct_dataframe(json_path, sep_audio_path, og_audio_path, cache_location):
    json_files = glob.glob(f"{json_path}/**/*.json", recursive=True)

    hash_file = hashlib.sha224(" ".join(json_files).encode('utf-8')).hexdigest()
    hash_file = os.path.join(os.path.join(cache_location, hash_file))
    os.makedirs(cache_location, exist_ok=True)
    print(f"Writing or looking for {hash_file}")

    if os.path.exists(hash_file):
        with open(hash_file, 'rb') as f:
            df = pickle.load(f)
            return df

    df = nussl.evaluation.aggregate_score_files(json_files)
    df = df[df['source'] == 'vocals']
    
    separated_files = glob.glob(
        f"{sep_audio_path}/**/*", recursive=True
    )

    separated_path_dict = {}
    for x in separated_files:
        k1 = os.path.splitext(x.split('/')[-1])[0]
        k2 = os.path.splitext(k1)[0]
        if k1 not in separated_path_dict:
            separated_path_dict[k1] = []
        if k2 not in separated_path_dict:
            separated_path_dict[k2] = []
        separated_path_dict[k1].append(x)
        separated_path_dict[k2].append(x)

    original_files = glob.glob(
         f"{og_audio_path}/**/*", recursive=True
    )
    
    def add_info_to_row(row):
        key = os.path.splitext(os.path.splitext(row['file'])[0])[0]
        song_name = re.findall(r"(?<= - ).*", key)[0]
        original_path = [x for x in original_files if song_name in x]
        try:
            label = original_path[0].split('/')[-2]
        except:
            label = 'unknown'
        if key in separated_path_dict:
            separated_path = separated_path_dict[key]
        else:
            separated_path = None
            label = 'unknown'
        confidence = row['posterior_confidence'] * row['silhouette_confidence']
        row['separated_path'] = separated_path
        row['original_path'] = original_path
        row['label'] = label
        row['confidence'] = confidence
        return row
    
    df = df.progress_apply(
        lambda row: add_info_to_row(row), axis=1
    )
    with open(hash_file, 'wb') as f:
        pickle.dump(df, f)
    return df

@gin.configurable
def filter_row_by_confidence(row, df, perc_min, perc_max):
    min_ = np.percentile(df['confidence'].dropna(), perc_min)
    max_ = np.percentile(df['confidence'].dropna(), perc_max)
    if row['confidence'] >= min_ and row['confidence'] < max_:
        return True
    return False

@gin.configurable
def filter_df_by_label(df, labels=None):
    if labels is None:
        return df
    mask = df['label'].isin(labels)
    df = df[mask]
    return df

LABELS = [
    'train', 
    'valid', 
    'rock', 
    'decades', 
    'classical', 
    'pop', 
    'pop2',
    'jazz',
    'opera',
    'oldies',
    'violin_concertos',
]
@gin.configurable
def construct_dataframe_from_json(json_path):
    json_files = glob.glob(f"{json_path}/**/*.json", recursive=True)
    df = nussl.evaluation.aggregate_score_files(json_files)
    df['confidence'] = df['silhouette_confidence'] * df['posterior_confidence']

    file_dict = {os.path.basename(x): x for x in json_files}

    def get_metadata(row):
        idx = 1 if row['source'] == 'vocals' else 0
        json_path = file_dict[row['file']]
        with open(json_path, 'r') as f:
            data = json.load(f)
        row['original_path'] = data['metadata']['original_path']
        row['separated_path'] = data['metadata']['separated_path'][idx]
        row['label'] = 'unknown'
        for label in LABELS:
            if label in row['original_path']:
                row['label'] = label
                break
        return row
        
    df = df.progress_apply(lambda x: get_metadata(x), axis=1)
    return df

@gin.configurable
def construct_symlink_folder(destination, df, rules):
    shutil.rmtree(destination, ignore_errors=True)
    os.makedirs(destination, exist_ok=True)

    df = filter_df_by_label(df)

    def symlink_row(row):
        accepted = True
        for rule in rules:
            if not rule(row, df):
                accepted = False
        if accepted:
            #symlink the files
            p = row['separated_path']
            fdir, fname = p.split('/')[-2:]
            os.makedirs(os.path.join(destination, fdir), exist_ok=True)
            d = os.path.join(destination, fdir, fname)
            os.symlink(p, d)
        
    df.progress_apply(
        lambda row: symlink_row(row), axis=1
    )
