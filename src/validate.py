import os
num_numpy_threads = '8'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads

import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import time
from rich import print

from data_utils import squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from utils import extract_index_features, collate_fn, device, get_model_path


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset,
                            blip_model: torch.nn.Module, 
                            index_features: torch.tensor, index_features_normed_pooled: torch.tensor,
                            index_names: List[str]) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param blip_model: stage 1 model
    :param index_features: validation index features
    :param index_features_normed_pooled: validation index features, pooled to 256 and normalized
    :param index_names: validation index names
    :return: the computed validation metrics
    """

    # Generate predictions with reference image and text
    predicted_features, target_names = generate_fiq_val_predictions(blip_model, relative_val_dataset,
                                                                    index_names, index_features)

    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    # Normalize the target candidates' index features
    index_features = index_features_normed_pooled.float() # already normed

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    recall_at60 = (torch.sum(labels[:, :60]) / len(labels)).item() * 100
    recall_at70 = (torch.sum(labels[:, :70]) / len(labels)).item() * 100
    recall_at80 = (torch.sum(labels[:, :80]) / len(labels)).item() * 100
    recall_at90 = (torch.sum(labels[:, :90]) / len(labels)).item() * 100
    recall_at100 = (torch.sum(labels[:, :100]) / len(labels)).item() * 100
    recall_at150 = (torch.sum(labels[:, :150]) / len(labels)).item() * 100
    recall_at200 = (torch.sum(labels[:, :200]) / len(labels)).item() * 100
    recall_at300 = (torch.sum(labels[:, :300]) / len(labels)).item() * 100
    recall_at400 = (torch.sum(labels[:, :400]) / len(labels)).item() * 100
    recall_at500 = (torch.sum(labels[:, :500]) / len(labels)).item() * 100

    if 'SAVE_TOPK' in globals():
        if SAVE_TOPK:
            print(f"{relative_val_dataset.dress_types} {relative_val_dataset.split} \nat50: {recall_at50} \nat60: {recall_at60} \nat70: {recall_at70} \nat80: {recall_at80} \nat90: {recall_at90} \nat100: {recall_at100} \nat150: {recall_at150} \nat200: {recall_at200} \nat300: {recall_at300} \nat400: {recall_at400} \nat500: {recall_at500}")
            breakpoint() # confirm saving to file

            dress_types = ','.join(relative_val_dataset.dress_types) # dset.triplets are fixed, directly loaded from json file. should be fine.
            save_dir = os.path.dirname(os.path.dirname(STAGE1_PATH)) + f'/fiq_top_{K_VALUE}_{relative_val_dataset.split}_{dress_types}.pt'
            torch.save({
                'sorted_index_names': sorted_index_names[:,:K_VALUE],
                'target_names': target_names,
                'index_names': index_names,
                'labels': labels[:,:K_VALUE],
                'split': relative_val_dataset.split,
                'dress_types': dress_types,
            }, save_dir)
            print(f"top {K_VALUE} saved at {save_dir}.")
    else: # SAVE_TOPK not defined if in training
        pass
    
    return recall_at10, recall_at50


def generate_fiq_val_predictions(blip_model: torch.nn.Module,
                                 relative_val_dataset: FashionIQDataset,
                                 index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param blip_model: stage1 model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                     num_workers=8, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features and target names
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    target_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):  # Load data

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(batch_target_names) == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = blip_model.img_txt_fusion(reference_image_features, None, input_captions, train=False)

        predicted_features = torch.vstack((predicted_features, batch_predicted_features)) # already normed
        target_names.extend(batch_target_names)

    return predicted_features, target_names


def fashioniq_val_retrieval(dress_type: str, model: torch.nn.Module, preprocess: callable, train=False):
    """
    Perform retrieval on FashionIQ validation set computing the metrics.
    :param dress_type: FashionIQ category on which perform the retrieval
    :param model: stageI model
    :param preprocess: preprocess pipeline
    """

    model = model.float().eval()

    # Define the validation datasets and extract the index features
    if not train:
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
        index_features, index_features_p, index_names = extract_index_features(classic_val_dataset, model, blip_stage1=True)
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
    else:
        classic_val_dataset = FashionIQDataset('train', [dress_type], 'classic', preprocess, force_validate=True)
        index_features, index_features_p, index_names = extract_index_features(classic_val_dataset, model, blip_stage1=True)
        relative_val_dataset = FashionIQDataset('train', [dress_type], 'relative', preprocess, force_validate=True)

    return compute_fiq_val_metrics(relative_val_dataset, model, index_features, index_features_p, 
                                   index_names)


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, 
                             blip_model: torch.nn.Module, 
                             index_features: torch.tensor, index_features_normed_pooled: torch.tensor,
                             index_names: List[str]) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param blip_model: stageI model
    :param index_features: validation index features
    :param index_features_normed_pooled: validation index features after normalization and pooling; if not using this feature then pass in the vanilla index features
    :param index_names: validation index names
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_features, reference_names, target_names, group_members = \
        generate_cirr_val_predictions(blip_model, relative_val_dataset, index_names, index_features)

    print(f"[{datetime.now()}] Compute CIRR validation metrics")

    # Normalize the target candidates' index features
    print(f"[{datetime.now()}] Compute the index features")
    index_features = index_features_normed_pooled.float() # already normed

    # Compute the distances and sort the results
    print(f"[{datetime.now()}] Compute the distances and sort the results")
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(target_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the ground-truth labels wrt the predictions
    print(f"[{datetime.now()}] Compute the ground-truth labels wrt the predictions")
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names) - 1).reshape(len(target_names), -1))

    # Compute the subset predictions and ground-truth labels
    print(f"[{datetime.now()}] Compute subset predictions and ground-truth labels")
    group_members = np.array(group_members)
    print(f"[{datetime.now()}] Compute group_mask")
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    print(f"[{datetime.now()}] Compute group_labels")
    group_labels = labels[group_mask].reshape(labels.shape[0], -1)

    print(f"[{datetime.now()}] Compute assert torch.equal")
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
    assert torch.equal(torch.sum(group_labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    print(f"[{datetime.now()}] Compute metrics")
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    recall_at60 = (torch.sum(labels[:, :60]) / len(labels)).item() * 100
    recall_at70 = (torch.sum(labels[:, :70]) / len(labels)).item() * 100
    recall_at80 = (torch.sum(labels[:, :80]) / len(labels)).item() * 100
    recall_at90 = (torch.sum(labels[:, :90]) / len(labels)).item() * 100
    recall_at100 = (torch.sum(labels[:, :100]) / len(labels)).item() * 100
    recall_at150 = (torch.sum(labels[:, :150]) / len(labels)).item() * 100
    recall_at200 = (torch.sum(labels[:, :200]) / len(labels)).item() * 100
    recall_at300 = (torch.sum(labels[:, :300]) / len(labels)).item() * 100
    recall_at400 = (torch.sum(labels[:, :400]) / len(labels)).item() * 100
    recall_at500 = (torch.sum(labels[:, :500]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100

    if 'SAVE_TOPK' in globals():
        if SAVE_TOPK:
            print(f"{relative_val_dataset.split} \nat50: {recall_at50} \nat60: {recall_at60} \nat70: {recall_at70} \nat80: {recall_at80} \nat90: {recall_at90} \nat100: {recall_at100} \nat150: {recall_at150} \nat200: {recall_at200} \nat300: {recall_at300} \nat400: {recall_at400} \nat500: {recall_at500}")
            breakpoint() # confirm saving to file

            save_dir = os.path.dirname(os.path.dirname(STAGE1_PATH)) + f'/cirr_top_{K_VALUE}_{relative_val_dataset.split}.pt'
            # by default save to the experiment folder
            torch.save({
                'sorted_index_names': sorted_index_names[:,:K_VALUE],
                'target_names': target_names,
                'index_names': index_names,
                'labels': labels[:,:K_VALUE],
                'group_labels': group_labels,
                'split': relative_val_dataset.split,
            }, save_dir)
            print(f"top {K_VALUE} saved at {save_dir}.")
    else: # SAVE_TOPK not defined if in training
        pass

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(blip_model: torch.nn.Module, 
                                  relative_val_dataset: CIRRDataset,
                                  index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param blip_model: stage1 model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print(f"[{datetime.now()}] Compute CIRR validation predictions")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    target_names = []
    group_members = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members in tqdm(
            relative_val_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(batch_target_names) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = blip_model.img_txt_fusion(reference_image_features, None, captions, train=False)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        target_names.extend(batch_target_names)
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)

    return predicted_features, reference_names, target_names, group_members


def cirr_val_retrieval(model: torch.nn.Module, preprocess: callable, train=False):
    """
    Perform retrieval on CIRR validation set computing the metrics.
    :param model: stage1 model
    :param preprocess: preprocess pipeline
    """

    model = model.float().eval()

    # Define the validation datasets and extract the index features
    if train:
        classic_val_dataset = CIRRDataset('train', 'classic', preprocess, force_validate=True)
        index_features, index_features_p, index_names = extract_index_features(classic_val_dataset, model, blip_stage1=True)
        relative_val_dataset = CIRRDataset('train', 'relative', preprocess, force_validate=True)
    else:
        classic_val_dataset = CIRRDataset('val', 'classic', preprocess)
        index_features, index_features_p, index_names = extract_index_features(classic_val_dataset, model, blip_stage1=True)
        relative_val_dataset = CIRRDataset('val', 'relative', preprocess)

    return compute_cirr_val_metrics(relative_val_dataset, model, index_features, index_features_p,
                                    index_names, )


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--stage1-path", type=str, help="path to trained Stage1")

    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['squarepad', 'targetpad'] ")
    
    parser.add_argument("--train", dest="train", action='store_true',
                        help="Whether to validate on train-split")
    
    parser.add_argument("--save-topk", dest="save_topk", action='store_true',
                        help="Whether to save the topK results")
    parser.add_argument("--k", default=200, type=int,
                        help="top-k value to be saved. For Fashion-IQ: default 100; for CIRR: default 50.")
    args = parser.parse_args()

    args.stage1_path = get_model_path(args.stage1_path, args.dataset.lower(), stage1=True)

    global SAVE_TOPK
    SAVE_TOPK = args.save_topk
    print(f'save_topk set to {SAVE_TOPK}')
    if SAVE_TOPK:
        global K_VALUE
        K_VALUE = int(args.k)
        global STAGE1_PATH
        STAGE1_PATH = args.stage1_path

    input_dim = 384

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.preprocess == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        pass

    from blip_stage1 import blip_stage1
    import yaml
    config = yaml.load(open('configs/retrieval_coco.yaml', 'r'), Loader=yaml.Loader)
    model = blip_stage1(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device, non_blocking=True)  
    state_dict = torch.load(args.stage1_path, map_location=device)
    model.load_state_dict(state_dict["BLIP_Retrieval"])
    model.eval()
    print('blip model checkpoint loaded successfully as Stage 1')

    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(model, preprocess, train=args.train)

        print(f"{group_recall_at1 = }")
        print(f"{group_recall_at2 = }")
        print(f"{group_recall_at3 = }")
        print(f"{recall_at1 = }")
        print(f"{recall_at5 = }")
        print(f"{recall_at10 = }")
        print(f"{recall_at50 = }")

        print(f"recall_mean ={(group_recall_at1 + recall_at5)/2}")

    elif args.dataset.lower() == 'fashioniq':
        average_recall10_list = []
        average_recall50_list = []

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', model,
                                                                     preprocess, train=args.train)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', model,
                                                                     preprocess, train=args.train)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', model,
                                                                       preprocess, train=args.train)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)

        print(f"\n{shirt_recallat10 = }")
        print(f"{shirt_recallat50 = }")

        print(f"{dress_recallat10 = }")
        print(f"{dress_recallat50 = }")

        print(f"{toptee_recallat10 = }")
        print(f"{toptee_recallat50 = }")

        print(f"average recall10 = {mean(average_recall10_list)}")
        print(f"average recall50 = {mean(average_recall50_list)}")

        print(f"average total = {(mean(average_recall50_list) + mean(average_recall10_list))/2}")
    else:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")


if __name__ == '__main__':
    main()
