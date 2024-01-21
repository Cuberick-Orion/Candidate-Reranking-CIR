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
from utils import extract_index_features, collate_fn, device

def get_model_path(model_path, dataset):
    '''
    helper function to obtain full model path

    Assume the actual checkpoint path to be like:
    (for FashionIQ)  models/<EXP_FOLDER_NAME>/saved_models/blip.pt
    (for CIRR)       models/<EXP_FOLDER_NAME>/saved_models/blip_mean.pt
    for both stageI and stageII.

    You can only provide the <EXP_FOLDER_NAME> string, this function can complete the rest.
    '''
    if 'models/' not in model_path[:7]:
        # prepend
        model_path = 'models/' + model_path
        assert os.path.exists(model_path), RuntimeError(f"case 0 model_path do not exists at {model_path}")
    if '.pt' not in model_path:
        # append
        if dataset == 'fashioniq':
            model_path = model_path + '/saved_models/blip.pt' 
        if dataset == 'cirr':
            model_path = model_path + '/saved_models/blip_mean.pt'
        assert os.path.exists(model_path), RuntimeError(f"case 1 model_path do not exists at {model_path}")
    else:
        # should be full path
        assert os.path.exists(model_path), RuntimeError(f"case 2 model_path do not exists at {model_path}")
    print(f"model path processed as {model_path}")
    return model_path

def get_top_k_path(exp_name, dataset):
    '''
    helper function to obtain full top-k path down to the pt file
    this function associates a pre-defined stageI experiment name with the top-k file path

    if no pre-defined associations are found, will assume the input string is the top-k file path and return it
    '''
    
    fiq_possible_top_ks = {
        'BLIP_stageI_b512_2e-5_cos20': 'deploy_ckpts/models/stage1/fashionIQ/fiq_top_200_val_DTYPE.pt',
    }
    cirr_possible_top_ks = {
        'BLIP_stageI_b512_2e-5_cos10': 'deploy_ckpts/models/stage1/CIRR/cirr_top_200_val.pt',
    }
    if exp_name is None:
        return None
    if dataset == 'fashioniq':
        try:
            return fiq_possible_top_ks[exp_name]
        except: # if no association, return raw
            assert os.path.exists(exp_name)
            return exp_name
    if dataset == 'cirr':
        try:
            return cirr_possible_top_ks[exp_name]
        except: # if no association, return raw
            assert os.path.exists(exp_name)
            return exp_name


def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, 
                            blip_model: torch.nn.Module, model_stage1: torch.nn.Module, 
                            index_features: torch.tensor,
                            index_names: List[str]) -> Tuple[float, float]:
    """
    Compute validation metrics on FashionIQ dataset
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param blip_model: stageII model
    :param index_features: validation index features
    :param index_features_tarProj: validation index features after tarProj; if not using tarProj then pass in the vanilla index features
    :param index_names: validation index names
    :return: the computed validation metrics
    """

    # Generate predictions
    predicted_logits, _ = generate_fiq_val_predictions(blip_model, model_stage1, relative_val_dataset,
                                                                    index_names, index_features) # N x K, logits the higher the better

    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    sorted_indices = torch.argsort(predicted_logits, dim=-1, descending=True).cpu() # the higher the better, sort high -> low


    labels = np.take_along_axis(relative_val_dataset.K_labels, sorted_indices.numpy(), axis=1) # N x K
    labels = torch.tensor(labels)

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    recall_at100 = (torch.sum(labels[:, :100]) / len(labels)).item() * 100
    print(f"{relative_val_dataset.dress_types}: recall_at100: {recall_at100:.2f}")
    
    
    return recall_at10, recall_at50


def generate_fiq_val_predictions(blip_model: torch.nn.Module, model_stage1: torch.nn.Module, 
                                 relative_val_dataset: FashionIQDataset,
                                 index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Compute FashionIQ predictions on the validation set
    :param blip_model: stageII model
    :param relative_val_dataset: FashionIQ validation dataset in relative mode
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features and target names
    """
    print(f"[{datetime.now()}] Compute FashionIQ {relative_val_dataset.dress_types} validation predictions")

    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=1,
                                     num_workers=8, pin_memory=True, collate_fn=collate_fn,
                                     shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))
    
    # Initialize predicted features and target names
    predicted_logits = torch.empty((0, relative_val_dataset.K)).to(device, non_blocking=True) # at the end, N x K
    target_names = []

    for reference_names, batch_target_names, captions, K_sorted_index_names, K_labels in tqdm(relative_val_loader):  # Load data
        if True in K_labels:
            # Concatenate the captions in a deterministic way
            flattened_captions: list = np.array(captions).T.flatten().tolist()
            input_captions = [
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
                i in range(0, len(flattened_captions), 2)]

            # Compute the predicted features
            with torch.no_grad():
                assert len(input_captions) == 1 # assume batch size is 1
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
                z_t = model_stage1.img_txt_fusion(reference_image_features, reference_image_features, input_captions, train=False, return_raw=True) # use ref_feat as placeholder for target here

                # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
                # a single tensor
                if relative_val_dataset.K == 1:
                    K_sorted_index_names = [i_[0] for i_ in K_sorted_index_names] # flatten the list from dataloader
                    candidate_image_features = itemgetter(*reference_names)(K_sorted_index_names).unsqueeze(0)
                else:
                    K_sorted_index_names = [i_[0] for i_ in K_sorted_index_names] # flatten the list from dataloader
                    candidate_image_features = torch.stack(itemgetter(*K_sorted_index_names)(
                        name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                
                batch_predicted_logits = blip_model.img_txt_fusion_val(z_t, candidate_image_features, input_captions) # K
                batch_predicted_logits = batch_predicted_logits.unsqueeze(0) # 1 x K
        else:
            # save time and don't do forward
            with torch.no_grad():
                batch_predicted_logits = torch.full((1, relative_val_dataset.K), -99999.99, device=device) # fill with very small values
        predicted_logits = torch.vstack((predicted_logits, batch_predicted_logits))
        target_names.extend(batch_target_names)
    
    print(f"{relative_val_dataset.dress_types[0]}: top-{relative_val_dataset.K} candidate {100 * relative_val_dataset.K_labels.sum() / relative_val_dataset.K_labels.shape[0]:.2f}%")

    return predicted_logits, target_names


def fashioniq_val_retrieval(dress_type: str, model: torch.nn.Module, model_stage1: torch.nn.Module, preprocess: callable):
    """
    Perform retrieval on FashionIQ validation set computing the metrics.
    :param dress_type: FashionIQ category on which perform the retrieval
    :param model: stageII model
    :param preprocess: preprocess pipeline
    """

    model = model.float().eval()
    model_stage1 = model_stage1.float().eval()

    # Define the validation datasets and extract the index features
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, load_topk=TOP_K_PATH.replace('DTYPE',dress_type), K=K_VALUE)
    index_features, index_names = extract_index_features(classic_val_dataset, model, blip_stage2=True)
    torch.cuda.empty_cache()
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, load_topk=TOP_K_PATH.replace('DTYPE',dress_type), K=K_VALUE)

    return compute_fiq_val_metrics(relative_val_dataset, model, model_stage1, index_features, 
                                   index_names)


def compute_cirr_val_metrics(relative_val_dataset: CIRRDataset, 
                             blip_model: torch.nn.Module, model_stage1: torch.nn.Module, 
                             index_features: torch.tensor,
                             index_names: List[str]) -> Tuple[
    float, float, float, float, float, float, float]:
    """
    Compute validation metrics on CIRR dataset
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param blip_model: stageII model
    :param index_features: validation index features
    :param index_features_p: validation index features after pooled and normalization
    :param index_names: validation index names
    :return: the computed validation metrics
    """
    # Generate predictions
    predicted_logits, group_predicted_logits, _, target_names, group_members = \
        generate_cirr_val_predictions(blip_model, model_stage1, 
                                      relative_val_dataset, index_names, index_features)

    print(f"[{datetime.now()}] Compute CIRR {relative_val_dataset.split}-split validation metrics")

    sorted_indices = torch.argsort(predicted_logits, dim=-1, descending=True).cpu() # the higher the better, sort high -> low

    print(f"[{datetime.now()}] Obtain the ground-truth labels wrt the predictions")

    labels = np.take_along_axis(relative_val_dataset.K_labels, sorted_indices.numpy(), axis=1) # N x K
    labels = torch.tensor(labels)

    # Compute the subset predictions and ground-truth labels
    # group_labels are computed following the steps in validate.py
    # association between group_members ---- group_predicted_logits
    # first obtain argsort index of the logits, then map to the group_members to obtain group_sorted_index_names
    # finally, obtain group_labels (T/F) by matching the target_names with the sorted index names
    print(f"[{datetime.now()}] Compute subset predictions and ground-truth labels")
    group_members = np.array(group_members); assert group_members.shape[1] == 5, ValueError()

    print(f"[{datetime.now()}] Obtain group_sorted_index_names and labels")
    group_sorted_indices = torch.argsort(group_predicted_logits, dim=-1, descending=True).cpu() # the higher the better, sort high -> low
    group_sorted_index_names = np.take_along_axis(group_members, group_sorted_indices.numpy(), axis=1) # N x K
    group_labels = torch.tensor( 
        group_sorted_index_names == np.repeat(np.array(target_names), 5).reshape(len(target_names), -1))

    # Compute the metrics
    recall_at1 = (torch.sum(labels[:, :1]) / len(labels)).item() * 100
    recall_at5 = (torch.sum(labels[:, :5]) / len(labels)).item() * 100
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    recall_at100 = (torch.sum(labels[:, :100]) / len(labels)).item() * 100
    group_recall_at1 = (torch.sum(group_labels[:, :1]) / len(group_labels)).item() * 100
    group_recall_at2 = (torch.sum(group_labels[:, :2]) / len(group_labels)).item() * 100
    group_recall_at3 = (torch.sum(group_labels[:, :3]) / len(group_labels)).item() * 100
    print(f"recall_at50: {recall_at50:.2f} recall_at100: {recall_at100:.2f} group_recall_at1: {group_recall_at1:.2f}")

    return group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50


def generate_cirr_val_predictions(blip_model: torch.nn.Module, model_stage1: torch.nn.Module, 
                                  relative_val_dataset: CIRRDataset,
                                  index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[str], List[List[str]]]:
    """
    Compute CIRR predictions on the validation set
    :param blip_model: stageII model
    :param relative_val_dataset: CIRR validation dataset in relative mode
    :param index_features: validation index features
    :param index_names: validation index names
    :return: predicted features, reference names, target names and group members
    """
    print(f"[{datetime.now()}] Compute CIRR validation predictions with K={relative_val_dataset.K}")
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=1, num_workers=8,
                                     pin_memory=True, collate_fn=collate_fn, shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize predicted features, target_names, group_members and reference_names
    predicted_logits = torch.empty((0, relative_val_dataset.K)).to(device, non_blocking=True)
    group_predicted_logits = torch.empty((0, 5)).to(device, non_blocking=True)
    target_names = []
    group_members_noRef = []
    reference_names = []

    for batch_reference_names, batch_target_names, captions, batch_group_members, K_sorted_index_names, K_labels, K_group_labels in tqdm(
            relative_val_loader):  # Load data
        K_sorted_index_names = [i_[0] for i_ in K_sorted_index_names]
        batch_group_members = np.array(batch_group_members).T.tolist()[0] # assume batch_size==1, full list of 6 (remove reference image in it later)
        if True in K_labels:
            # Compute the predicted features
            with torch.no_grad():
                assert len(captions) == 1 # assume batch size is 1
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
                z_t = model_stage1.img_txt_fusion(reference_image_features, None, captions, train=False, return_raw=True)

                # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
                # a single tensor
                if relative_val_dataset.K == 1:
                    candidate_image_features = itemgetter(*K_sorted_index_names)(name_to_feat).unsqueeze(0)
                else:
                    candidate_image_features = torch.stack(itemgetter(*K_sorted_index_names)(
                        name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
                
                batch_predicted_logits = blip_model.img_txt_fusion_val(z_t, candidate_image_features, captions)
                batch_predicted_logits = batch_predicted_logits.unsqueeze(0) # 1 x K
        else:
            with torch.no_grad():
                batch_predicted_logits = torch.full((1, relative_val_dataset.K), -99999.99, device=device) # fill with very small values
            
        #for group members
        with torch.no_grad():
            assert len(captions) == 1 # assume batch size is 1
            reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            z_t = model_stage1.img_txt_fusion(reference_image_features, None, captions, train=False, return_raw=True)

            group_candidate_image_features = torch.stack(itemgetter(*[ii for ii in batch_group_members if ii not in batch_reference_names])(
                name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            group_batch_predicted_logits = blip_model.img_txt_fusion_val(z_t, group_candidate_image_features, captions)
            group_batch_predicted_logits = group_batch_predicted_logits.unsqueeze(0) # 1 x 5

        predicted_logits = torch.vstack((predicted_logits, batch_predicted_logits))
        group_predicted_logits = torch.vstack((group_predicted_logits, group_batch_predicted_logits))
        target_names.extend(batch_target_names)
        group_members_noRef.append([ii for ii in batch_group_members if ii not in batch_reference_names]) # assume batch_size=1, the end result should be a nested list
        reference_names.extend(batch_reference_names)

    print(f"{relative_val_dataset.split}-split: top-{relative_val_dataset.K} candidate {100 * relative_val_dataset.K_labels.sum() / relative_val_dataset.K_labels.shape[0]:.2f}%")
    return predicted_logits, group_predicted_logits, reference_names, target_names, group_members_noRef


def cirr_val_retrieval(model: torch.nn.Module, model_stage1: torch.nn.Module, preprocess: callable):
    """
    Perform retrieval on CIRR validation set computing the metrics.
    :param model: stageII model
    :param preprocess: preprocess pipeline
    """

    model = model.float().eval() 
    model_stage1 = model_stage1.float().eval() 

    # Define the validation datasets and extract the index features
    classic_val_dataset = CIRRDataset('val', 'classic', preprocess, load_topk=TOP_K_PATH, K=K_VALUE)
    index_features, index_names = extract_index_features(classic_val_dataset, model, blip_stage2=True)
    torch.cuda.empty_cache()
    relative_val_dataset = CIRRDataset('val', 'relative', preprocess, load_topk=TOP_K_PATH, K=K_VALUE)

    return compute_cirr_val_metrics(relative_val_dataset, model, model_stage1, index_features,
                                    index_names)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")

    parser.add_argument("--stage1-path", type=str, help="path to trained stageI")
    parser.add_argument("--stage2-path", type=str, help="path to trained stageII")

    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['squarepad', 'targetpad'] ")

    parser.add_argument("--top-k-path", type=str, help="path to top-k")
    parser.add_argument("--k", default=50, type=int, help="top-k value to be used for validation, the larger the longer it takes")
    
    args = parser.parse_args()

    args.stage1_path = get_model_path(args.stage1_path, args.dataset.lower())
    args.stage2_path = get_model_path(args.stage2_path, args.dataset.lower())

    global TOP_K_PATH
    TOP_K_PATH = get_top_k_path(args.top_k_path, args.dataset.lower())
    print(f"Top-K pt file loaded at {TOP_K_PATH}")
    global K_VALUE
    K_VALUE = int(args.k)
    print(f"K value chosen as {K_VALUE}")

    input_dim = 384

    if args.transform == 'targetpad':
        print('Target pad preprocess pipeline is used')
        preprocess = targetpad_transform(args.target_ratio, input_dim)
    elif args.preprocess == 'squarepad':
        print('Square pad preprocess pipeline is used')
        preprocess = squarepad_transform(input_dim)
    else:
        pass

    # initialize a blip_retrieval
    from blip_stage1 import blip_stage1
    import yaml
    config = yaml.load(open('configs/retrieval_coco.yaml', 'r'), Loader=yaml.Loader)
    model_stage1 = blip_stage1(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model_stage1 = model_stage1.to(device) 

    print('Trying to load the stageI model')
    state_dict = torch.load(args.stage1_path, map_location=device)
    model_stage1.load_state_dict(state_dict["BLIP_Retrieval"])
    print('stageI model loaded successfully')
    model_stage1.eval()

    from blip_stage2 import blip_stage2
    config = yaml.load(open('configs/nlvr.yaml', 'r'), Loader=yaml.Loader)
    model = blip_stage2(pretrained=config['pretrained'], image_size=config['image_size'], med_config='configs/med_config.json',
                    vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model = model.to(device, non_blocking=True)  

    print('Trying to load the stageII model')
    state_dict = torch.load(args.stage2_path, map_location=device)
    model.load_state_dict(state_dict["BLIP_NLVR"])
    print('stageII model loaded successfully')
    model.eval()

    if args.dataset.lower() == 'cirr':
        group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = \
            cirr_val_retrieval(model, model_stage1, preprocess)

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

        shirt_recallat10, shirt_recallat50 = fashioniq_val_retrieval('shirt', model, model_stage1,
                                                                     preprocess)
        average_recall10_list.append(shirt_recallat10)
        average_recall50_list.append(shirt_recallat50)
        torch.cuda.empty_cache()

        dress_recallat10, dress_recallat50 = fashioniq_val_retrieval('dress', model, model_stage1,
                                                                     preprocess)
        average_recall10_list.append(dress_recallat10)
        average_recall50_list.append(dress_recallat50)
        torch.cuda.empty_cache()

        toptee_recallat10, toptee_recallat50 = fashioniq_val_retrieval('toptee', model, model_stage1,
                                                                     preprocess)
        average_recall10_list.append(toptee_recallat10)
        average_recall50_list.append(toptee_recallat50)
        torch.cuda.empty_cache()

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
