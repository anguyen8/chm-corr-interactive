import gdown
import torchvision
import os

# Embeddings
gdown.cached_download(
    url="https://drive.google.com/uc?id=116CiA_cXciGSl72tbAUDoN-f1B9Frp89",
    path="./embeddings.pickle",
    quiet=False,
    md5="002b2a7f5c80d910b9cc740c2265f058",
)

# labels
gdown.download(id="1SDtq6ap7LPPpYfLbAxaMGGmj0EAV_m_e")

# iNat Trained ResNet-50
gdown.cached_download(
    url="https://drive.google.com/uc?id=1_wVPVamytcZFS7FJWkNzz3XMQh95SHRe",
    path="BBN.iNaturalist2017.res50.90epoch.best_model.pth",
    quiet=False,
    md5="5cce3d554bc8e1e239b19321f692d9c5",
)

# CUB training set
gdown.cached_download(
    url="https://drive.google.com/uc?id=1iR19j7532xqPefWYT-BdtcaKnsEokIqo",
    path="./CUB_train.zip",
    quiet=False,
    md5="1bd99e73b2fea8e4c2ebcb0e7722f1b1",
)

# EXTRACT training set
# if data/train not exists then extract
if not os.path.exists("./data/train"):
    torchvision.datasets.utils.extract_archive(
        from_path="CUB_train.zip",
        to_path="data/",
        remove_finished=False,
    )

# CHM Weights
gdown.cached_download(
    url="https://drive.google.com/uc?id=1yM1zA0Ews2I8d9-BGc6Q0hIAl7LzYqr0",
    path="pas_psi.pt",
    quiet=False,
    md5="6b7b4d7bad7f89600fac340d6aa7708b",
)
