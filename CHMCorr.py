# CHM-Corr Classifier
import argparse
import json
import pickle
import random
from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from common.evaluation import Evaluator
from model import chmnet
from model.base.geometry import Geometry

from Utils import (
    CosineCustomDataset,
    PairedLayer4Extractor,
    compute_spatial_similarity,
    generate_mask,
    normalize_array,
    get_transforms,
    arg_topK,
)

# Setting the random seed
random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
to_np = lambda x: x.data.to("cpu").numpy()

# CHMNet Config
chm_args = dict(
    {
        "alpha": [0.05, 0.1],
        "img_size": 240,
        "ktype": "psi",
        "load": "pas_psi.pt",
    }
)


class CHMGridTransfer:
    def __init__(
        self,
        query_image,
        support_set,
        support_set_labels,
        train_folder,
        top_N,
        top_K,
        binarization_threshold,
        chm_source_transform,
        chm_target_transform,
        cosine_source_transform,
        cosine_target_transform,
        batch_size=64,
    ):
        self.N = top_N
        self.K = top_K
        self.BS = batch_size

        self.chm_source_transform = chm_source_transform
        self.chm_target_transform = chm_target_transform
        self.cosine_source_transform = cosine_source_transform
        self.cosine_target_transform = cosine_target_transform

        self.source_embeddings = None
        self.target_embeddings = None
        self.correspondence_map = None
        self.similarity_maps = None
        self.reverse_similarity_maps = None
        self.transferred_points = None

        self.binarization_threshold = binarization_threshold

        # UPDATE THIS
        self.q = query_image
        self.support_set = support_set
        self.labels_ss = support_set_labels

    def build(self):
        # C.M.H
        test_ds = CosineCustomDataset(
            query_image=self.q,
            supporting_set=self.support_set,
            source_transform=self.chm_source_transform,
            target_transform=self.chm_target_transform,
        )
        test_dl = DataLoader(test_ds, batch_size=self.BS, shuffle=False)
        self.find_correspondences(test_dl)

        # LAYER 4s
        test_ds = CosineCustomDataset(
            query_image=self.q,
            supporting_set=self.support_set,
            source_transform=self.cosine_source_transform,
            target_transform=self.cosine_target_transform,
        )
        test_dl = DataLoader(test_ds, batch_size=self.BS, shuffle=False)
        self.compute_embeddings(test_dl)
        self.compute_similarity_map()

    def find_correspondences(self, test_dl):
        model = chmnet.CHMNet(chm_args["ktype"]).to(device)
        model.load_state_dict(
            torch.load(chm_args["load"], map_location=torch.device("cpu"))
        )
        Evaluator.initialize(chm_args["alpha"])
        Geometry.initialize(img_size=chm_args["img_size"])

        grid_results = []
        transferred_points = []

        # FIXED GRID HARD CODED
        fixed_src_grid_points = list(
            product(
                np.linspace(1 + 17, 240 - 17 - 1, 7),
                np.linspace(1 + 17, 240 - 17 - 1, 7),
            )
        )
        fixed_src_grid_points = np.asarray(fixed_src_grid_points, dtype=np.float64).T

        with torch.no_grad():
            model.eval()
            for idx, batch in enumerate(tqdm(test_dl)):
                keypoints = (
                    torch.tensor(fixed_src_grid_points)
                    .unsqueeze(0)
                    .repeat(batch["src_img"].shape[0], 1, 1)
                ).to(device)

                n_pts = torch.tensor(
                    np.asarray(batch["src_img"].shape[0] * [49]), dtype=torch.long
                ).to(device)

                corr_matrix = model(
                    batch["src_img"].to(device), batch["trg_img"].to(device)
                )
                prd_kps = Geometry.transfer_kps(
                    corr_matrix, keypoints, n_pts, normalized=False
                )
                transferred_points.append(prd_kps.cpu().numpy())
                for tgt_points in prd_kps:
                    tgt_grid = []
                    for x, y in zip(tgt_points[0], tgt_points[1]):
                        tgt_grid.append(
                            [int(((x + 1) / 2.0) * 7), int(((y + 1) / 2.0) * 7)]
                        )
                    grid_results.append(tgt_grid)

        self.correspondence_map = grid_results
        self.transferred_points = np.vstack(transferred_points)

    def compute_embeddings(self, test_dl):
        paired_extractor = PairedLayer4Extractor()

        source_embeddings = []
        target_embeddings = []

        with torch.no_grad():
            for idx, batch in enumerate(test_dl):
                s_e, t_e = paired_extractor((batch["src_img"], batch["trg_img"]))

                source_embeddings.append(s_e)
                target_embeddings.append(t_e)

        # EMBEDDINGS
        self.source_embeddings = torch.cat(source_embeddings, axis=0)
        self.target_embeddings = torch.cat(target_embeddings, axis=0)

    def compute_similarity_map(self):
        CosSim = nn.CosineSimilarity(dim=0, eps=1e-6)

        similarity_maps = []
        rsimilarity_maps = []

        grid = []
        for i in range(7):
            for j in range(7):
                grid.append([i, j])

        # Compute for all image pairs
        for i in range(len(self.correspondence_map)):
            cosine_map = np.zeros((7, 7))
            reverse_cosine_map = np.zeros((7, 7))

            # calculate cosine based on the chm corr. map
            for S, T in zip(grid, self.correspondence_map[i]):
                v1 = self.source_embeddings[i][:, S[0], S[1]]
                v2 = self.target_embeddings[i][:, T[0], T[1]]
                covalue = CosSim(v1, v2)
                cosine_map[S[0], S[1]] = covalue
                reverse_cosine_map[T[0], T[1]] = covalue

            similarity_maps.append(cosine_map)
            rsimilarity_maps.append(reverse_cosine_map)

        self.similarity_maps = similarity_maps
        self.reverse_similarity_maps = rsimilarity_maps


    # def compute_score_using_cc(self):
    #     num_embeddings = len(self.source_embeddings)
    #     SIMS_source = np.zeros((num_embeddings, *self.source_embeddings[0].shape[1:]))

    #     for i in range(num_embeddings):
    #         SIMS_source[i], _ = compute_spatial_similarity(
    #             to_np(self.source_embeddings[i]), to_np(self.target_embeddings[i])
    #         )

    #     num_maps = len(self.similarity_maps)
    #     top_cos_values = np.zeros(num_maps)

    #     for i in range(num_maps):
    #         cosine_value = np.multiply(self.similarity_maps[i], generate_mask(normalize_array(SIMS_source[i]), t=self.binarization_threshold))
    #         reshaped_cosine_value = cosine_value.T.reshape(-1)
    #         top_5_indicies = np.argsort(reshaped_cosine_value)[::-1][:5]
    #         top_cos_values[i] = np.mean(reshaped_cosine_value[top_5_indicies])

    #     return top_cos_values.tolist()

    def compute_score_using_cc(self):
        # Assuming compute_spatial_similarity can handle batch processing
        SIMS_source, _ = compute_spatial_similarity(
            to_np(self.source_embeddings), to_np(self.target_embeddings)
        )

        cosine_values = np.multiply(self.similarity_maps, 
                                    generate_mask(normalize_array(SIMS_source), 
                                                t=self.binarization_threshold))

        reshaped_cosine_values = cosine_values.transpose(0, 2, 1).reshape(cosine_values.shape[0], -1)
        top_5_indicies = np.argsort(reshaped_cosine_values)[:, -5:]
        top_cos_values = np.mean(np.take_along_axis(reshaped_cosine_values, top_5_indicies, axis=1), axis=1)

        return top_cos_values.tolist()


    # def compute_score_using_cc(self):
    #     # CC MAPS
    #     SIMS_source, SIMS_target = [], []
    #     for i in range(len(self.source_embeddings)):
    #         simA, simB = compute_spatial_similarity(
    #             to_np(self.source_embeddings[i]), to_np(self.target_embeddings[i])
    #         )

    #         SIMS_source.append(simA)
    #         SIMS_target.append(simB)

    #     SIMS_source = np.stack(SIMS_source, axis=0)
    #     # SIMS_target = np.stack(SIMS_target, axis=0)

    #     top_cos_values = []

    #     for i in range(len(self.similarity_maps)):
    #         cosine_value = np.multiply(
    #             self.similarity_maps[i],
    #             generate_mask(
    #                 normalize_array(SIMS_source[i]), t=self.binarization_threshold
    #             ),
    #         )
    #         top_5_indicies = np.argsort(cosine_value.T.reshape(-1))[::-1][:5]
    #         mean_of_top_5 = np.mean(
    #             [cosine_value.T.reshape(-1)[x] for x in top_5_indicies]
    #         )
    #         top_cos_values.append(np.mean(mean_of_top_5))

    #     return top_cos_values

    # def compute_score_using_custom_points(self, input_mask):
    #     top_cos_values = []

    #     for i in range(len(self.similarity_maps)):
    #         cosine_value = np.multiply(self.similarity_maps[i], input_mask)
    #         top_indicies = np.argsort(cosine_value.T.reshape(-1))[::-1]
    #         mean_of_tops = np.mean(
    #             [cosine_value.T.reshape(-1)[x] for x in top_indicies]
    #         )
    #         top_cos_values.append(np.mean(mean_of_tops))

    #     return top_cos_values

    def compute_score_using_custom_points(self, selected_keypoint_masks):
        num_maps = len(self.similarity_maps)
        top_cos_values = np.zeros(num_maps)

        for i in range(num_maps):
            cosine_value = np.multiply(self.similarity_maps[i], selected_keypoint_masks)
            self.similarity_maps_masked.append(cosine_value)
            
            reshaped_cosine_value = cosine_value.T.reshape(-1)
            top_indicies = np.argsort(reshaped_cosine_value)[::-1]
            mean_of_tops = np.mean(reshaped_cosine_value[top_indicies])
            top_cos_values[i] = mean_of_tops

        return top_cos_values.tolist()


    def export(self):
        storage = {
            "N": self.N,
            "K": self.K,
            "source_embeddings": self.source_embeddings,
            "target_embeddings": self.target_embeddings,
            "correspondence_map": self.correspondence_map,
            "similarity_maps": self.similarity_maps,
            "T": self.binarization_threshold,
            "query": self.q,
            "support_set": self.support_set,
            "labels_for_support_set": self.labels_ss,
            "rsimilarity_maps": self.reverse_similarity_maps,
            "transferred_points": self.transferred_points,
        }

        return ModifiableCHMResults(storage)


class ModifiableCHMResults:
    def __init__(self, storage):
        self.N = storage["N"]
        self.K = storage["K"]
        self.source_embeddings = storage["source_embeddings"]
        self.target_embeddings = storage["target_embeddings"]
        self.correspondence_map = storage["correspondence_map"]
        self.similarity_maps = storage["similarity_maps"]
        self.T = storage["T"]
        self.q = storage["query"]
        self.support_set = storage["support_set"]
        self.labels_ss = storage["labels_for_support_set"]
        self.rsimilarity_maps = storage["rsimilarity_maps"]
        self.transferred_points = storage["transferred_points"]
        self.similarity_maps_masked = None
        self.SIMS_source = None
        self.SIMS_target = None
        self.masked_sim_values = []
        self.top_cos_values = []
        self.score_cache = {}
        self.all_scores = None
        self.test = 1
        self.top_cos_values_cc = None
        self.top_cos_values_custom = {}


    def compute_score_using_cc(self):
        # CC MAPS
        SIMS_source, SIMS_target = [], []
        for i in range(len(self.source_embeddings)):
            simA, simB = compute_spatial_similarity(
                to_np(self.source_embeddings[i]), to_np(self.target_embeddings[i])
            )

            SIMS_source.append(simA)
            SIMS_target.append(simB)

        SIMS_source = np.stack(SIMS_source, axis=0)
        SIMS_target = np.stack(SIMS_target, axis=0)

        self.SIMS_source = SIMS_source
        self.SIMS_target = SIMS_target

        top_cos_values = []

        for i in range(len(self.similarity_maps)):
            masked_sim_values = np.multiply(
                self.similarity_maps[i],
                generate_mask(normalize_array(SIMS_source[i]), t=self.T),
            )
            self.masked_sim_values.append(masked_sim_values)
            top_5_indicies = np.argsort(masked_sim_values.T.reshape(-1))[::-1][:5]
            mean_of_top_5 = np.mean(
                [masked_sim_values.T.reshape(-1)[x] for x in top_5_indicies]
            )
            top_cos_values.append(np.mean(mean_of_top_5))

        self.top_cos_values = top_cos_values

        return top_cos_values

    def compute_score_using_custom_points(self, selected_keypoint_masks):
        top_cos_values = []
        similarity_maps_masked = []

        for i in range(len(self.similarity_maps)):
            cosine_value = np.multiply(self.similarity_maps[i], selected_keypoint_masks)
            similarity_maps_masked.append(cosine_value)
            top_indicies = np.argsort(cosine_value.T.reshape(-1))[::-1]
            mean_of_tops = np.mean(
                [cosine_value.T.reshape(-1)[x] for x in top_indicies]
            )
            top_cos_values.append(np.mean(mean_of_tops))

        self.similarity_maps_masked = similarity_maps_masked
        return top_cos_values

    def predict_using_cc(self, K=None):
        K = K or self.K 
        
        if self.top_cos_values_cc is None:
            self.top_cos_values_cc = self.compute_score_using_cc()


        top_cos_values = self.top_cos_values_cc
        
        # Predict
        prediction = np.argmax(
            np.bincount(
                [self.labels_ss[x] for x in np.argsort(top_cos_values)[::-1][: K]]
            )
        )
        prediction_weight = np.max(
            np.bincount(
                [self.labels_ss[x] for x in np.argsort(top_cos_values)[::-1][: K]]
            )
        )

        reranked_nns_idx = [x for x in np.argsort(top_cos_values)[::-1]]
        reranked_nns_files = [self.support_set[x] for x in reranked_nns_idx]

        topK_idx = [
            x
            for x in np.argsort(top_cos_values)[::-1]
            if self.labels_ss[x] == prediction
        ]
        topK_files = [self.support_set[x] for x in topK_idx]
        topK_cmaps = [self.correspondence_map[x] for x in topK_idx]
        topK_similarity_maps = [self.similarity_maps[x] for x in topK_idx]
        topK_rsimilarity_maps = [self.rsimilarity_maps[x] for x in topK_idx]
        topK_transfered_points = [self.transferred_points[x] for x in topK_idx]
        predicted_folder_name = topK_files[0].split("/")[-2]

        return (
            topK_idx,
            prediction,
            predicted_folder_name,
            prediction_weight,
            topK_files[: K],
            reranked_nns_files[: K],
            topK_cmaps[: K],
            topK_similarity_maps[: K],
            topK_rsimilarity_maps[: K],
            topK_transfered_points[: K],
        )

    def predict_using_custom_mask(self, selected_keypoint_masks, K=None):
        K = K or self.K 
        
        # convert selected_keypoint_masks to value using a hash
        hash_key =  hash(selected_keypoint_masks.tostring())

        if hash_key not in self.top_cos_values_custom.keys():
            self.top_cos_values_custom[hash_key] = self.compute_score_using_custom_points(selected_keypoint_masks)

        top_cos_values = self.top_cos_values_custom[hash_key]

        # Predict
        prediction = np.argmax(
            np.bincount(
                [self.labels_ss[x] for x in np.argsort(top_cos_values)[::-1][: K]]
            )
        )
        prediction_weight = np.max(
            np.bincount(
                [self.labels_ss[x] for x in np.argsort(top_cos_values)[::-1][: K]]
            )
        )

        reranked_nns_idx = [x for x in np.argsort(top_cos_values)[::-1]]
        reranked_nns_files = [self.support_set[x] for x in reranked_nns_idx]

        topK_idx = [
            x
            for x in np.argsort(top_cos_values)[::-1]
            if self.labels_ss[x] == prediction
        ]
        topK_files = [self.support_set[x] for x in topK_idx]
        topK_cmaps = [self.correspondence_map[x] for x in topK_idx]
        topK_similarity_maps = [self.similarity_maps[x] for x in topK_idx]
        topK_rsimilarity_maps = [self.rsimilarity_maps[x] for x in topK_idx]
        topK_transferred_points = [self.transferred_points[x] for x in topK_idx]
        # topK_scores = [top_cos_values[x] for x in topK_idx]
        topK_masked_sims = [self.similarity_maps_masked[x] for x in topK_idx]
        predicted_folder_name = topK_files[0].split("/")[-2]

        non_zero_mask = np.count_nonzero(selected_keypoint_masks)

        return (
            topK_idx,
            prediction,
            predicted_folder_name,
            prediction_weight,
            topK_files[: K],
            reranked_nns_files[: K],
            topK_cmaps[: K],
            topK_similarity_maps[: K],
            topK_rsimilarity_maps[: K],
            topK_transferred_points[: K],
            topK_masked_sims[: K],
            non_zero_mask,
        )


def export_visualizations_results(
    reranker_output,
    knn_predicted_label,
    knn_confidence,
    topK_knns,
    K,
    N,
    T=0.55,
    mask=None,
):
    """
    Export all details for visualization and analysis
    """
    if mask is None:
        non_zero_mask = 5  # default value
        (
            topK_idx,
            p,
            pfn,
            pr,
            rfiles,
            reranked_nns,
            cmaps,
            sims,
            rsims,
            trns_kpts,
        ) = reranker_output.predict_using_cc(K=K)
    else:
        (
            topK_idx,
            p,
            pfn,
            pr,
            rfiles,
            reranked_nns,
            cmaps,
            sims,
            rsims,
            trns_kpts,
            tok_k_msk_sims,
            non_zero_mask,
        ) = reranker_output.predict_using_custom_mask(mask, K=K)

    if mask is None:
        MASKED_COSINE_VALUES = [
            np.multiply(
                sims[X],
                generate_mask(
                    normalize_array(reranker_output.SIMS_source[topK_idx[X]]), t=T
                ),
            )
            for X in range(len(sims))
        ]
    else:
        MASKED_COSINE_VALUES = [np.multiply(sims[X], mask) for X in range(len(sims))]

    list_of_source_points = []
    list_of_target_points = []

    for CK in range(len(sims)):
        target_keypoints = []
        topk_index = arg_topK(MASKED_COSINE_VALUES[CK], topK=non_zero_mask)

        for i in range(non_zero_mask):  # Number of Connections
            # Psource = point_list[topk_index[i]]
            x, y = trns_kpts[CK].T[topk_index[i]]
            Ptarget = int(((x + 1) / 2.0) * 240), int(((y + 1) / 2.0) * 240)
            target_keypoints.append(Ptarget)

        # Uniform Grid of points
        a = np.linspace(1 + 17, 240 - 17 - 1, 7)
        b = np.linspace(1 + 17, 240 - 17 - 1, 7)
        point_list = list(product(a, b))

        list_of_source_points.append(np.asarray([point_list[x] for x in topk_index]))
        list_of_target_points.append(np.asarray(target_keypoints))

    # EXPORT OUTPUT
    detailed_output = {
        "q": reranker_output.q,
        "K": K,
        "N": N,
        "knn-prediction": knn_predicted_label,
        "knn-prediction-confidence": knn_confidence,
        "knn-nearest-neighbors": topK_knns,
        "chm-prediction": pfn,
        "chm-prediction-confidence": pr,
        "chm-nearest-neighbors": rfiles,
        "chm-nearest-neighbors-all": reranked_nns,
        "correspondance_map": cmaps,
        "masked_cos_values": MASKED_COSINE_VALUES,
        "src-keypoints": list_of_source_points,
        "tgt-keypoints": list_of_target_points,
        "non_zero_mask": non_zero_mask,
        "transferred_kpoints": trns_kpts,
    }

    return detailed_output


def chm_cache_results(
    query_image, kNN_results, support, TRAIN_SET, N, K, T=0.55, BS=64
):
    global chm_args
    chm_src_t, chm_tgt_t, cos_src_t, cos_tgt_t = get_transforms("single", chm_args)
    knn_predicted_label, knn_confidence, topK_knns = kNN_results

    reranker = CHMGridTransfer(
        query_image=query_image,
        support_set=support[0],
        support_set_labels=support[1],
        train_folder=TRAIN_SET,
        top_N=N,
        top_K=K,
        binarization_threshold=T,
        chm_source_transform=chm_src_t,
        chm_target_transform=chm_tgt_t,
        cosine_source_transform=cos_src_t,
        cosine_target_transform=cos_tgt_t,
        batch_size=BS,
    )

    # Building the reranker
    reranker.build()
    # Make a ModifiableCHMResults
    exported_reranker = reranker.export()
    return exported_reranker


def chm_classify_and_visualize(
    query_image, kNN_results, support, TRAIN_SET, N, K, T=0.55, BS=64
):
    knn_predicted_label, knn_confidence, topK_knns = kNN_results

    exported_reranker = chm_cache_results(
        query_image, kNN_results, support, TRAIN_SET, N, K, T, BS
    )
    output = export_visualizations_results(
        exported_reranker,
        knn_predicted_label,
        knn_confidence,
        topK_knns,
        K,
        N,
        T,
    )

    return output


def chm_classify_from_cache_CC_visualize(kNN_results, chm_results, N, K, T=0.55, BS=64):
    mask = None

    knn_predicted_label, knn_confidence, topK_knns = kNN_results
    output = export_visualizations_results(
        chm_results, knn_predicted_label, knn_confidence, topK_knns, K, N, T, mask=mask
    )

    return output


def chm_classify_from_cache_masked_visualize(
    kNN_results, chm_results, mask, N, K, T=0.55, BS=64
):
    # if mask is all zero, pass None
    is_all_zero = np.count_nonzero(mask) == 0
    if is_all_zero:
        mask = None

    knn_predicted_label, knn_confidence, topK_knns = kNN_results
    output = export_visualizations_results(
        chm_results, knn_predicted_label, knn_confidence, topK_knns, K, N, T, mask=mask
    )

    return output
