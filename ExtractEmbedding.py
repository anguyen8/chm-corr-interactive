import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from FeatureExtractors import resnet50_features


to_np = lambda x: x.data.to("cpu").numpy()
device = "cuda" if torch.cuda.is_available() else "cpu"


def QueryToEmbedding(query, device=device):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Load the pretrained ResNet50 model
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = resnet50.to(device)
    resnet50.eval()

    # Remove the last two layers (avgpool and fc) to get the Layer 4 output
    resnet50_layer4 = torch.nn.Sequential(*(list(resnet50.children())[:-1]))
    # Preprocess the input image
    query_pil = Image.open(query).convert("RGB")
    image_pt = preprocess(query_pil).unsqueeze(0).to(device)

    # Extract the layer 4 embedding
    with torch.inference_mode():
        embedding = resnet50_layer4(image_pt)

    embedding = embedding.view(-1)
    embedding = to_np(embedding)

    return np.asarray([embedding])


def QueryToEmbeddingiNat(query, device=device):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Load the pretrained ResNet50 model
    model = resnet50_features(inat=True, pretrained=True)
    model.eval()
    model.to(device)

    # Preprocess the input image
    query_pil = Image.open(query).convert("RGB")
    image_pt = preprocess(query_pil).unsqueeze(0).to(device)

    # Extract the layer 4 embedding
    with torch.inference_mode():
        embedding = model(image_pt)
        embedding = F.adaptive_avg_pool2d(embedding, 1).view(embedding.size(0), -1)

    embedding = embedding.view(-1)
    embedding = to_np(embedding)

    return np.asarray([embedding])
