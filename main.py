import ast
import csv
import io
import json
import json as js
import os
import pickle
import random
import string
import sys
import tempfile
import time
from collections import Counter
from glob import glob
import pandas as pd
import gdown
import gradio as gr
import numpy as np
import requests
import torch
import torchvision
import torchvision.transforms.functional as F
from fastapi.staticfiles import StaticFiles
from huggingface_hub import HfApi, login, snapshot_download
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from CHMCorr import (
    chm_cache_results,
    chm_classify_from_cache_CC_visualize,
    chm_classify_from_cache_masked_visualize,
)
from ExtractEmbedding import QueryToEmbedding, QueryToEmbeddingiNat
from fileservice import app
from SimSearch import SearchableTrainingSet
from visualization import plot_from_reranker_corrmap

import gptanalysis as gpta

# Constants
PICKLE_READ_MODE = "rb"
JPG_FORMAT = "jpg"
TRAIN_DIR_PATH = "./data/train/"
CHM_CACHE_PATH = "./cache/*_chmcache.pickle"
EMBEDDINGS_PATH = "./embeddings_iNat.pickle"
LABELS_PATH = "./labels.pickle"
USER_INTERACTIONS_PATH = "user_interactions.jsonl"
USER_DECISIONS_PATH = "user_decisions.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FINAL_QUESTION = """ <p> Is the image <strong><span style="font-size: 1.2em;">`{}`</span></strong>? Having seen the <strong><span style="font-size: 1.2em;">AI's original prediction</span></strong> and its explanation, and having made additional predictions using the manipulation tool, you can now make a final decision about <em><span style="font-size: 1.2em;">the AI's original prediction</span></em> (panel on the top).</p>"""

TOTAL_SAMPLES = 20

Files = [
    "./cache/Hooded_Warbler_0007_164633_chmcache.pickle",
    "./cache/Baird_Sparrow_0001_794578_chmcache.pickle",
]

confidence_mapping = {
    "1 Not confident at all": 1,
    "2 Not very confident": 2,
    "3 Neither": 3,
    "4 Fairly confident": 4,
    "5 Very confident": 5,
}


# Login into the huggingface
LOG_HF_TOKEN = "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"

# Global Variables
cached_samples = list(sorted(glob(CHM_CACHE_PATH)))
csv.field_size_limit(sys.maxsize)

preprocess = transforms.Compose(
    transforms=[transforms.Resize(256), transforms.CenterCrop(240)]
)


# Ugly Global Variables
ACCEPT_RECLASSIFICATION_MESSAGE = "Accept the Reclassification"
REJECT_RECLASSIFICATION_MESSAGE = "Not Sure!"


# Helper Functions
def load_pickle(file_path):
    with open(file_path, PICKLE_READ_MODE) as f:
        return pickle.load(f)


get_window_url_params = """
    function(text_input, url_params) {
        console.log(text_input, url_params);
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return [text_input, url_params];
        }
    """

def check_if_user_file_exists(username, data_type):
    data = read_data_from_hub(username, data_type)
    print("previous data", data)
    return bool(data)


def read_data_from_hub(username, data_type):
    url = f"https://huggingface.co/datasets/luulinh90s/chm-corr-prj-giang/raw/main/{username}_{data_type}.jsonl"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while trying to fetch data: {e}")
        return {}

    try:
        data = [json.loads(item) for item in response.text.split("\n") if item]
        return data
    except json.JSONDecodeError as e:
        print(f"An error occurred while trying to decode the data: {e}")
        return {}


def write_record_to_dataset(record, username, data_type):
    global LOG_HF_TOKEN
    if check_if_user_file_exists(username, data_type):
        data = read_data_from_hub(username, data_type)
        data.append(record)
    else:
        data = [record]

    temp_filename = f"{username}_{data_type}.jsonl"
    with open(temp_filename, "w") as outfile:
        for entry in data:
            json.dump(entry, outfile)
            outfile.write("\n")

    hf_api = HfApi(token=LOG_HF_TOKEN)

    hf_api.upload_file(
        path_or_fileobj=temp_filename,
        path_in_repo=f"{username}_{data_type}.jsonl",
        repo_id="luulinh90s/chm-corr-prj-giang",
        repo_type="dataset",
    )
    os.remove(temp_filename)


def write_interaction_to_file(
    current_user, current_index, user_mask, chm_predictions, random_seed
):
    top_1 = sorted(chm_predictions.items(), key=lambda x: x[1], reverse=True)[0][0]
    interaction_data = {
        "username": current_user["username"],
        "image_index": current_index,
        "user_mask": user_mask.tolist(),
        "chm_predictions": chm_predictions,
        "chm_predictions_top_1": top_1,
        "random_seed": int(random_seed),
    }
    log_data_to_file("user_interactions.jsonl", interaction_data)
    write_record_to_dataset(interaction_data, current_user["username"], "interactions")


def write_decision_to_file(
    current_user, current_index, decision, ai_prediction, confidence, random_seed
):
    decision_data = {
        "username": current_user["username"],
        "gt-label": current_user["gt-labels"][current_index],
        "ai-prediction": ai_prediction,
        "image-index": current_index,
        "decision": decision,
        "self-reported-confidence": confidence_mapping.get(confidence, -1),
        "user-random-set": current_user["samples"],
        "random_seed": int(random_seed),
    }
    log_data_to_file("user_decisions.jsonl", decision_data)
    write_record_to_dataset(decision_data, current_user["username"], "decisions")
    return decision_data


def log_data_to_file(filename, data):
    with open(filename, "a") as f:
        json.dump(data, f)
        f.write("\n")


custom_css = """
        #container {
            position: relative;
            width: 400px;
            height: 400px;
            border: 1px solid #000;
        }
        #canvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
"""

html_text = """
    <div id="container">
        <canvas id="canvas" width="512" height="512"></canvas><img id="canvas-background" style="display:none;"/>
    </div>
"""

with open(f"./embeddings_iNat.pickle", "rb") as f:
    Xtrain = pickle.load(f)

with open(f"./labels.pickle", "rb") as f:
    ytrain = pickle.load(f)


# Extract label names
training_folder = ImageFolder(root="./data/train/")
id_to_bird_name = {
    x[1]: x[0].split("/")[-2].replace(".", " ") for x in training_folder.imgs
}


def parse_bool_string(s):
    try:
        bool_list = ast.literal_eval(s)
        if not isinstance(bool_list, list):
            raise ValueError("The input string must represent a list.")
        return bool_list
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Invalid input string: {e}")


def make_custom_classification(current_user, current_index, json_data, random_seed):
    print("current index: ", current_index)

    with open(current_user["samples"][current_index], "rb") as f:
        cache_data = pickle.load(f)
        fname = current_user["samples"][current_index]
        print(f"loaded cache from: {fname}")

    kNN_results = cache_data["knn_cache"]
    chm_cache = cache_data["chm_cache"]
    query_image = cache_data["hidden_image"]
    K = cache_data["K"]
    N = cache_data["N"]

    N = int(N)
    K = int(K)

    user_mask = np.asarray(parse_bool_string(json_data), np.int32)

    chm_output = chm_classify_from_cache_masked_visualize(
        kNN_results, chm_cache, user_mask, N=N, K=K
    )

    chm_output_labels = Counter(
        [x.split("/")[-2] for x in chm_output["chm-nearest-neighbors-all"][:K]]
    )

    max_count = chm_output_labels.most_common(1)[0][1]
    top_predictions = [
        item for item in chm_output_labels.items() if item[1] == max_count
    ]

    print(f"top_predictions: {top_predictions}")

    final_K = K
    while len(top_predictions) != 1 and K < N:
        K += 1
        print("++++++++============= Breaking ties ++++++++=============")
        print(f"K: {K}")

        chm_output = chm_classify_from_cache_masked_visualize(
            kNN_results, chm_cache, user_mask, N=N, K=K
        )
        chm_output_labels = Counter(
            [x.split("/")[-2] for x in chm_output["chm-nearest-neighbors-all"][:K]]
        )
        max_count = chm_output_labels.most_common(1)[0][1]
        top_predictions = [
            item for item in chm_output_labels.items() if item[1] == max_count
        ]
    final_K = K

    final_prediction = top_predictions[0] if top_predictions else None

    human_readable = {x.replace(".", " "): y for x, y in chm_output_labels.items()}
    write_interaction_to_file(
        current_user, current_index, user_mask, human_readable, random_seed
    )

    fig, chm_output_label = plot_from_reranker_corrmap(chm_output)

    # Resize the visualization

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format=JPG_FORMAT)
    image = Image.open(img_buf)
    width, height = image.size
    new_width = width
    new_height = height

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    viz_image = image.crop((left + 310, top + 60, right - 248, bottom - 80))

    return (
        viz_image,
        {l: s / final_K for l, s in human_readable.items()},
        gr.update(interactive=True),  # For accept_btn
        gr.update(interactive=True),  # For reject_btn
        gr.update(interactive=True),  # For confidence_radio
    )


def default_classification(current_user, current_index, json_data):
    if current_index >= TOTAL_SAMPLES:
        # TODO Show final accuracy
        return (
            gr.update(interactive=False),  # For accept_btn
            gr.update(interactive=False),  # For reject_btn
            gr.update(interactive=False),  # For reclassify_btn
            None,
            None,
            None,
            None,
            None,
        )

    print(current_user["samples"])
    for i, item in enumerate(current_user["samples"]):
        print(i, item)

    with open(current_user["samples"][current_index], "rb") as f:
        cache_data = pickle.load(f)
        print(f"loaded cache from: {current_user['samples'][current_index]}")

    kNN_results = cache_data["knn_cache"]
    chm_cache = cache_data["chm_cache"]
    query_image = cache_data["hidden_image"]
    K = cache_data["K"]
    N = cache_data["N"]

    print(f"making classification: {N}, {K}")
    N = int(N)
    K = int(K)

    user_mask = np.asarray(parse_bool_string(json_data), np.int32)

    chm_output = chm_classify_from_cache_CC_visualize(kNN_results, chm_cache, N=N, K=K)

    chm_output_labels = Counter(
        [x.split("/")[-2] for x in chm_output["chm-nearest-neighbors-all"][:K]]
    )

    max_count = chm_output_labels.most_common(1)[0][1]
    top_predictions = [
        item for item in chm_output_labels.items() if item[1] == max_count
    ]

    print(f"top_predictions: {top_predictions}")

    final_K = K
    while len(top_predictions) != 1 and K < N:
        K += 1
        print("++++++++============= Breaking ties ++++++++=============")
        print(f"K: {K}")
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

    final_prediction = top_predictions[0] if top_predictions else None

    human_readable = {x.replace(".", " "): y for x, y in chm_output_labels.items()}

    question_text = FINAL_QUESTION.format(final_prediction[0])

    fig, chm_output_label = plot_from_reranker_corrmap(chm_output)

    # Resize the visualization

    img_buf = io.BytesIO()
    fig.savefig(img_buf, format=JPG_FORMAT)
    image = Image.open(img_buf)
    width, height = image.size
    new_width = width
    new_height = height

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    viz_image = image.crop((left + 310, top + 60, right - 248, bottom - 80))

    return (
        gr.update(interactive=True),  # For accept_btn
        gr.update(interactive=True),  # For reject_btn
        gr.update(interactive=True),  # For reclassify_btn
        viz_image,
        {l: s / final_K for l, s in human_readable.items()},
        {},
        question_text,
        gr.update(interactive=True, value=0),  # For confidence_radio
    )


def start_experiment_fn(username_txt, current_user, random_seed):
    # shuffle 10 samples for this user
    global cached_samples

    # set seed to the time
    seed_val = int(random_seed)
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)

    # Shuffle the samples
    cached_samples = list(sorted(glob(CHM_CACHE_PATH)))
    random.shuffle(cached_samples)

    df = pd.read_csv("./newset_600.csv", header=0)
    df["cache_file"] = df["cache_file"].apply(lambda x: x.split("/")[-1])

    correct_preds = []
    wrong_preds = []

    for c in cached_samples:
        c_name = c.split("/")[-1]
        matching_row = df[df["cache_file"] == c_name]

        if matching_row.empty:
            print(f"No matching row found for: {c_name}")
            continue

        if matching_row["correct"].values[0] == 1:
            print(f"Processing: {c_name} -- Correct classification")
            correct_preds.append(c)
        else:
            wrong_preds.append(c)
            print(f"Processing: {c_name} -- Wrong classification")

    print(f"len(correct_preds): {len(correct_preds)}")
    print(f"len(wrong_preds): {len(wrong_preds)}")

    random_set = random.sample(correct_preds, TOTAL_SAMPLES // 2) + random.sample(
        wrong_preds, TOTAL_SAMPLES // 2
    )
    random.shuffle(random_set)

    gt_labels = {}
    with open("./gt_labels_for_cache.json", "r") as f:
        gt_labels = json.load(f)

    random_set_labels = [gt_labels[x.split("/")[-1]] for x in random_set]

    current_user = {
        "username": username_txt,
        "samples": random_set,
        "gt-labels": random_set_labels,
    }

    # for i, item in enumerate(random_set):
    #     print(f"sample {i}: {item}")

    current_index = 0
    with open(current_user["samples"][current_index], "rb") as f:
        cache_data = pickle.load(f)
        fname = current_user["samples"][current_index]
        print(f"loaded cache from: {fname}")

    knn_cache = cache_data["knn_cache"]
    chm_cache = cache_data["chm_cache"]
    query_image = cache_data["hidden_image"]
    K = cache_data["K"]
    N = cache_data["N"]

    # return and ready to go
    return (
        current_user,
        current_index,
        query_image,
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def disable_accept_reject():
    return (
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def load_next_sample(
    current_user, current_index, user_choice_btn, ai_prediction, confidence, random_seed
):
    if current_user == "N/A":
        raise gr.Error("Please enter a username")

    _ = write_decision_to_file(
        current_user,
        current_index,
        user_choice_btn,
        ai_prediction,
        confidence,
        random_seed,
    )

    # load the next one
    new_index = current_index + 1

    if new_index >= TOTAL_SAMPLES:
        # TODO Calculate the accuracy

        # Get the json and calculate the accuracy
        # decision_data
        username = current_user["username"]
        data_type = "decisions"
        url = f"https://huggingface.co/datasets/luulinh90s/chm-corr-prj-giang/raw/main/{username}_{data_type}.jsonl"
        analysis_json, analysis_text = gpta.compute_accuracies_from_data(url)

        return (
            new_index,
            Image.open("./final.png"),
            new_index,
            None,
            None,
            analysis_text,
        )

    with open(current_user["samples"][new_index], "rb") as f:
        cache_data = pickle.load(f)

    knn_cache = cache_data["knn_cache"]
    chm_cache = cache_data["chm_cache"]
    query_image = cache_data["hidden_image"]
    K = cache_data["K"]
    N = cache_data["N"]

    return (
        new_index,
        query_image,
        new_index,
        None,
        None,
        None,
    )


def check_confidence(confidence_radio):
    conf = confidence_mapping.get(confidence_radio, -1)
    if conf == -1:
        raise gr.Error("Please select a confidence level")


def enable_componments():
    return gr.update(interactive=True)


def disable_componments():
    return gr.update(interactive=False)


def append_random_string_to_userna(input_username):
    # set random seed to the time
    random.seed(time.time())
    random_str = "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(8)
    )
    return input_username + "-" + random_str + "-DEBUG"


def predict(text, url_params):
    print(url_params)
    return ["Hello " + text + "!!", url_params]


with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="CHM-Cor++") as demo:
    # load images from the cache directory

    with gr.Column():
        title_text = gr.Markdown(
            "# Human Study - CHM-Corr++ - `Accept / Reject`"
        )
        gr.Markdown(
            "- In this study, you will evaluate decisions of an Artificial Intelligence (AI) system that can identify bird species from images. For each query image, you will see the AI’s decision along with other information, and decide whether to accept or reject the AI’s decision."
        )
        gr.Markdown(
            "- There are three parts to the study. In part 1, you will see the AI’s original decision and explanations. In part 2, you will use an `attention manipulation tool` that allows you to guide the AI’s attention and see if and how the AI updates its decision. Based on your explorations, in part 3, you will accept or reject the AI’s original decision."
        )

    current_index = gr.State(-1)
    current_user = gr.State("N/A")

    with gr.Column():
        with gr.Row():
            username_txt = gr.Textbox(label="Username", value=f"user")
            random_seed_txt = gr.Textbox(label="Random Seed", value="42")
            comepleted_images_cntr_txt = gr.Textbox(
                label="Completed Queries", value="0"
            )
            total_images = gr.Textbox(label="Total Queries", value=TOTAL_SAMPLES)

        prepare_experiment_btn = gr.Button(value="Start Experiment")

    gr.HTML("<br>")

    with gr.Column():
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Part 1: AI’s original decision")
                gr.Markdown(
                    "In the following panel, you can see the query image along with AI's decision and its explanation."
                )
                query_preview = gr.Image(type="filepath", visible=True)

            with gr.Column():
                ai_original = gr.Label(label="AI Decision")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "These are `some` of the bird species the AI considers the most likely along with the AI’s confidence for each option."
            )
            result_viz_ai_only = gr.Image(type="pil", label="Explanations").style(
                height="auto"
            )

    gr.HTML("<br>")

    with gr.Column():
        with gr.Column():
            gr.Markdown(
                "## Part 2: Explore the AI’s behavior by manipulating its attention"
            )
            gr.Markdown(
                "Use the following tool to guide the AI’s attention and see if and how the AI updates its decision."
            )

    reclassification_group = gr.Group(visible=True)
    with reclassification_group:
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "There is a 7x7 grid on top of the query image. Click on each grid cell to either select or deselect it. The selected grid cells will be the AI’s new attention (regions of the image to focus on when making the decision). Next, click on the “Make decision” button to see if and how the AI updates its decision."
                )
                html = gr.HTML(html_text)

            with gr.Column():
                reclassify_btn = gr.Button(
                    "Make decision based on the new attention", interactive=False
                )
                bridge_text = gr.Textbox("", visible=False)
                pred_label = gr.Label(label="Human+AI Decision")

        with gr.Row():
            hidden_image = gr.Image(type="pil", visible=False)

        with gr.Row():
            result_viz_reclassification = gr.Image(
                type="pil", label="Explanations"
            ).style(height="auto")

    gr.HTML("<br>")

    with gr.Column():
        with gr.Column():
            gr.Markdown("# Part 3: Your final decision")
            html_question = gr.HTML("")

        with gr.Row():
            accept_btn = gr.Button(value="Accept", interactive=False)
            reject_btn = gr.Button(value="Reject", interactive=False)

        with gr.Column():
            gr.Markdown("## Confidence Level")
            with gr.Row():
                confidence_radio = gr.Radio(
                    [
                        "1 Not confident at all",
                        "2 Not very confident",
                        "3 Neither",
                        "4 Fairly confident",
                        "5 Very confident",
                    ],
                    label="Confidence Level",
                    interactive=False,
                    default="3",
                )

    gr.HTML("<br>")
    with gr.Column():
        gr.Markdown("# Part 4. Final Report")
        with gr.Column():
            json_report = gr.JSON()
            json_raw = gr.Textbox(label="Accuracy results")





    # gr.HTML("<br>")
    # gr.Markdown("## Debugging --- Please ignore the below section")
    # url_params = gr.JSON({}, visible=True, label="URL Params")
    # text_input = gr.Text(label="Input")
    # text_output = gr.Text(label="Output")

    # btn = gr.Button("Run")
    # btn.click(fn=predict, inputs=[text_input, url_params], outputs=[text_output, url_params], _js=get_window_url_params)

    # ---------------------------------- Callbacks ---------------------------------- #

    prepare_experiment_btn.click(
        fn=append_random_string_to_userna, inputs=[username_txt], outputs=[username_txt]
    ).then(
        start_experiment_fn,
        inputs=[username_txt, current_user, random_seed_txt],
        outputs=[
            current_user,
            current_index,
            query_preview,
            username_txt,
            prepare_experiment_btn,
            random_seed_txt,
        ],
    ).then(
        fn=default_classification,
        inputs=[current_user, current_index, bridge_text],
        outputs=[
            accept_btn,
            reject_btn,
            reclassify_btn,
            result_viz_ai_only,
            ai_original,
            pred_label,
            html_question,
            confidence_radio,
        ],
        _js="(current_user, CHMPATH, BTEXT) => { return [current_user, CHMPATH, read_js_Data()]; }",
    )

    query_preview.change(
        fn=None,
        inputs=[query_preview],
        outputs=[],
        _js="(image) => { initializeEditor(); importBackground(image); return []; }",
    )

    # ---------------------------------- Reclassify Callbacks ---------------------------------- #

    reclassify_btn.click(
        fn=disable_accept_reject,
        inputs=[],
        outputs=[accept_btn, reject_btn, confidence_radio],
    ).then(
        fn=make_custom_classification,
        inputs=[current_user, current_index, bridge_text, random_seed_txt],
        outputs=[
            result_viz_reclassification,
            pred_label,
            accept_btn,
            reject_btn,
            confidence_radio,
        ],
        _js="(current_user, current_index, bridge_text, random_seed_txt) => { return [current_user, current_index, read_js_Data(), random_seed_txt]; }",
    )

    # ---------------------------------- Accept Callbacks ---------------------------------- #
    accept_btn.click(fn=check_confidence, inputs=[confidence_radio]).success(
        fn=disable_componments, inputs=[], outputs=[accept_btn]
    ).then(fn=disable_componments, inputs=[], outputs=[reject_btn]).then(
        fn=disable_componments, inputs=[], outputs=[confidence_radio]
    ).then(
        fn=disable_componments, inputs=[], outputs=[reclassify_btn]
    ).then(
        fn=load_next_sample,
        inputs=[
            current_user,
            current_index,
            accept_btn,
            ai_original,
            confidence_radio,
            random_seed_txt,
        ],
        outputs=[
            current_index,
            query_preview,
            comepleted_images_cntr_txt,
            result_viz_reclassification,
            json_report,
            json_raw,
        ],
    ).then(
        fn=default_classification,
        inputs=[current_user, current_index, bridge_text],
        outputs=[
            accept_btn,
            reject_btn,
            reclassify_btn,
            result_viz_ai_only,
            ai_original,
            pred_label,
            html_question,
            confidence_radio,
        ],
        _js="(current_user, CHMPATH, BTEXT) => { return [current_user, CHMPATH, read_js_Data()]; }",
    ).then(
        fn=None,
        _js="() => {  clear_grid(); return []; }",
    )

    # ---------------------------------- Reject Callbacks ---------------------------------- #

    reject_btn.click(fn=check_confidence, inputs=[confidence_radio]).success(
        fn=disable_componments, inputs=[], outputs=[accept_btn]
    ).then(fn=disable_componments, inputs=[], outputs=[reject_btn]).then(
        fn=disable_componments, inputs=[], outputs=[confidence_radio]
    ).then(
        fn=disable_componments, inputs=[], outputs=[reclassify_btn]
    ).then(
        fn=load_next_sample,
        inputs=[
            current_user,
            current_index,
            reject_btn,
            ai_original,
            confidence_radio,
            random_seed_txt,
        ],
        outputs=[
            current_index,
            query_preview,
            comepleted_images_cntr_txt,
            result_viz_reclassification,
            json_report,
            json_raw,
        ],
    ).then(
        fn=default_classification,
        inputs=[current_user, current_index, bridge_text],
        outputs=[
            accept_btn,
            reject_btn,
            reclassify_btn,
            result_viz_ai_only,
            ai_original,
            pred_label,
            html_question,
            confidence_radio,
        ],
        _js="(current_user, CHMPATH, BTEXT) => { return [current_user, CHMPATH, read_js_Data()]; }",
    ).then(
        fn=None,
        _js="() => {  clear_grid(); return []; }",
    )

app.mount("/js", StaticFiles(directory="js"), name="js")
gr.mount_gradio_app(app, demo, path="/")
