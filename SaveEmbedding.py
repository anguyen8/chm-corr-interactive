import time
import os
import torch
import numpy as np
import torchvision
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle
import argparse


concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.to("cpu").numpy()


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.model = model
        self.avgpool_output = None
        self.query = None
        self.cossim_value = {}

        def fw_hook(module, input, output):
            self.avgpool_output = output.squeeze()

        self.model.avgpool.register_forward_hook(fw_hook)

    def forward(self, input):
        _ = self.model(input)
        return self.avgpool_output

    def __repr__(self):
        return "Wrappper"


def run(training_set_path):
    # Standard ImageNet Transformer to apply imagenet's statistics to input batch
    dataset_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    training_imagefolder = ImageFolder(
        root=training_set_path, transform=dataset_transform
    )
    train_loader = torch.utils.data.DataLoader(
        training_imagefolder,
        batch_size=512,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    print(f"# of Training folder samples: {len(training_imagefolder)}")
    ########################################################################################################################
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    myw = Wrapper(model)

    training_embeddings = []
    training_labels = []

    with torch.no_grad():
        for _, (data, target) in enumerate(tqdm(train_loader)):
            embeddings = to_np(myw(data))
            labels = to_np(target)

            training_embeddings.append(embeddings)
            training_labels.append(labels)

    training_embeddings_concatted = concat(training_embeddings)
    training_labels_concatted = concat(training_labels)
    
    print(training_embeddings_concatted.shape)
    return training_embeddings_concatted, training_labels_concatted


def main():
    parser = argparse.ArgumentParser(description="Saving Embeddings")
    parser.add_argument("--train", help="Path to the Dataaset", type=str, required=True)
    args = parser.parse_args()

    embeddings, labels = run(args.train)

    # Caluclate Accuracy
    with open(f"embeddings.pickle", "wb") as f:
        pickle.dump(embeddings, f)

    with open(f"labels.pickle", "wb") as f:
        pickle.dump(labels, f)


if __name__ == "__main__":
    main()
