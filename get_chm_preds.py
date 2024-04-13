import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from CHMCorr import chm_classify_from_cache_CC_visualize
from glob import glob
from tqdm import tqdm

# Constants
PICKLE_READ_MODE = "rb"

# Load cache file
def load_cache_file(file_path):
    with open(file_path, PICKLE_READ_MODE) as f:
        return pickle.load(f)

# Default classification based on cache file
def default_classification_from_cache(cache_file):
    cache_data = load_cache_file(cache_file)
    kNN_results = cache_data["knn_cache"]
    chm_cache = cache_data["chm_cache"]
    K = cache_data["K"]
    N = cache_data["N"]

    N = int(N)
    K = int(K)

    chm_output = chm_classify_from_cache_CC_visualize(kNN_results, chm_cache, N=N, K=K)

    chm_output_labels = Counter(
        [x.split("/")[-2] for x in chm_output["chm-nearest-neighbors-all"][:K]]
    )

    max_count = chm_output_labels.most_common(1)[0][1]
    top_predictions = [
        item for item in chm_output_labels.items() if item[1] == max_count
    ]

    # Store the original top predictions before breaking the tie
    original_top_predictions = dict(top_predictions)

    final_K = K
    while len(top_predictions) != 1 and K < N:
        K += 1
        chm_output = chm_classify_from_cache_CC_visualize(
            kNN_results, chm_cache, N=N, K=K
        )
        chm_output_labels = Counter(
            [x.split("/")[-2] for x in chm_output["chm-nearest-neighbors-all"][:K]]
        )
        max_count = chm_output_labels.most_common(1)[0][1]
        top_predictions = [
            item for item in chm_output_labels.items() if item[1] == max_count
        ]
    final_K = K

    final_prediction = top_predictions[0][0] if top_predictions else None

    # Check if the prediction is correct
    correct_prediction = 1 if gt_labels[cache_file] == get_species_name(final_prediction) else 0

    return pd.DataFrame({
        'cache_file': [cache_file],
        'original_top_predictions': [original_top_predictions],
        'final_top_predictions': [dict(top_predictions)],
        'final_prediction': [final_prediction],
        'initial_K': [cache_data["K"]],
        'final_K': [final_K],
        'correct': [correct_prediction]
    })

def get_species_name(name):
    # Extracting the species name after the initial numbers and before the underscores
    return name.split('.')[-1].replace('_', ' ')

# Use the functions
cache_files = list(glob("./cache/*.pickle"))

# Get ground truth labels
gt_labels = {x: x.split('/')[-1][: x.split('/')[-1].find(next(filter(str.isdigit, x.split('/')[-1]))) - 1].replace('_', ' ') for x in cache_files}

df_list = []
for cache_file in tqdm(cache_files):
    prediction_df = default_classification_from_cache(cache_file)
    df_list.append(prediction_df)

final_df = pd.concat(df_list, ignore_index=True)
print(final_df)

final_df.to_csv("./newset_600.csv", index=False)
