import os, socket
'''
Manually limiting the thread number for numpy
this is recommended if your CPU has many threads
'''
num_numpy_threads = '8'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads

import json
import multiprocessing
from argparse import ArgumentParser
from operator import itemgetter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import CIRRDataset, targetpad_transform, squarepad_transform, base_path
from utils import extract_index_features, device, get_model_path, get_top_k_path


def generate_cirr_test_submissions(model: torch.nn.Module, model_stage1: torch.nn.Module, preprocess: callable, file_name: str):
    """
   Generate and save CIRR test submission files to be submitted to evaluation server
   :param model: stageII model
   :param preprocess: preprocess pipeline
   """
    model = model.float().eval()
    model_stage1 = model_stage1.float().eval() 

    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset('test1', 'classic', preprocess, load_topk=TOP_K_PATH, K=K_VALUE)
    index_features, index_names = extract_index_features(classic_test_dataset, model, blip_stage2=True)
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess, load_topk=TOP_K_PATH, K=K_VALUE)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset,
                                                                                  model, model_stage1,
                                                                                  index_features,
                                                                                  index_names)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_predictions)
    group_submission.update(pairid_to_group_predictions)

    # Define submission path
    submissions_folder_path = base_path / "submission" / 'CIRR'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    print(f"Saving CIRR test predictions")
    with open(submissions_folder_path / f"recall_submission_{file_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"recall_subset_submission_{file_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def generate_cirr_test_dicts(relative_test_dataset: CIRRDataset, 
                             blip_model: torch.nn.Module, model_stage1: torch.nn.Module, 
                             index_features: torch.tensor,
                             index_names: List[str]) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param blip_model: stageII model
    :param index_features: test index features
    :param index_names: test index names
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """
    # Generate predictions
    predicted_logits, group_predicted_logits, _, group_members, pairs_id = \
        generate_cirr_test_predictions(blip_model, model_stage1, relative_test_dataset, index_names, index_features)

    print(f"Compute CIRR prediction dicts")

    sorted_indices = torch.argsort(predicted_logits, dim=-1, descending=True).cpu() # the higher the better, sort high -> low
    sorted_index_names = np.take_along_axis(relative_test_dataset.K_sorted_index_names, sorted_indices.numpy(), axis=1)

    # Compute the subset predictions
    group_members = np.array(group_members); assert group_members.shape[1] == 5, ValueError()
    group_sorted_indices = torch.argsort(group_predicted_logits, dim=-1, descending=True).cpu() # the higher the better, sort high -> low
    sorted_group_names = np.take_along_axis(group_members, group_sorted_indices.numpy(), axis=1) # N x K


    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(blip_model: torch.nn.Module, model_stage1: torch.nn.Module,
                                   relative_test_dataset: CIRRDataset,
                                   index_names: List[str], index_features: torch.tensor) -> \
        Tuple[torch.tensor, List[str], List[List[str]], List[str]]:
    """
    Compute CIRR predictions on the test set
    :param blip_model: stage1 model
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param index_features: test index features
    :param index_names: test index names

    :return: predicted_features, reference_names, group_members and pairs_id
    """
    print(f"Compute CIRR test predictions")
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=1,
                                      num_workers=8, pin_memory=True, shuffle=False)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_logits = torch.empty((0, relative_test_dataset.K)).to(device, non_blocking=True)
    group_predicted_logits = torch.empty((0, 5)).to(device, non_blocking=True)
    group_members_noRef = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members, K_sorted_index_names in tqdm(
            relative_test_loader):  # Load data
        K_sorted_index_names = [i_[0] for i_ in K_sorted_index_names]
        batch_group_members = np.array(batch_group_members).T.tolist()[0] # assume batch_size==1, full list of 6 (remove reference image in it later)

        # Compute the predicted features
        with torch.no_grad():
            assert len(captions) == 1 # assume batch size is 1
            reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unsqueeze(0)
            z_t = model_stage1.img_txt_fusion(reference_image_features, None, captions, train=False, return_raw=True)
            
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if relative_test_dataset.K == 1:
                candidate_image_features = itemgetter(*K_sorted_index_names)(name_to_feat).unsqueeze(0)
            else:
                candidate_image_features = torch.stack(itemgetter(*K_sorted_index_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            
            batch_predicted_logits = blip_model.img_txt_fusion_val(z_t, candidate_image_features, captions)
            batch_predicted_logits = batch_predicted_logits.unsqueeze(0) # 1 x K

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
        group_members_noRef.append([ii for ii in batch_group_members if ii not in batch_reference_names]) # assume batch_size=1, the end result should be a nested list
        reference_names.extend(batch_reference_names)

        pairs_id.extend(batch_pairs_id)

    return predicted_logits, group_predicted_logits, reference_names, group_members_noRef, pairs_id


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="submission file name")

    parser.add_argument("--stage1-path", type=str, help="path to trained Stage1")
    parser.add_argument("--stage2-path", type=str, help="path to trained Stage2")

    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['squarepad', 'targetpad'] ")

    parser.add_argument("--top-k-path", type=str, help="path to top-k")
    parser.add_argument("--k", default=50, type=int,
                        help="top-k value.")
    args = parser.parse_args()

    args.stage1_path = get_model_path(args.stage1_path, 'cirr')
    args.stage2_path = get_model_path(args.stage2_path, 'cirr')

    global TOP_K_PATH
    TOP_K_PATH = get_top_k_path(args.top_k_path, 'cirr')
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

    from blip_stage1 import blip_stage1
    import yaml
    config = yaml.load(open('configs/retrieval_coco.yaml', 'r'), Loader=yaml.Loader)
    model_stage1 = blip_stage1(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model_stage1 = model_stage1.to(device, non_blocking=True)  

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


    generate_cirr_test_submissions(model, model_stage1, preprocess, args.submission_name)


if __name__ == '__main__':
    main()
