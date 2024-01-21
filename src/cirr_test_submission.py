import os, socket
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
from utils import extract_index_features, device, get_model_path


def generate_cirr_test_submissions(model: torch.nn.Module, preprocess: callable, file_name: str):
    """
   Generate and save CIRR test submission files to be submitted to evaluation server
   :param model: stage1 model
   :param preprocess: preprocess pipeline
   """
    model = model.float().eval()

    # Define the dataset and extract index features
    classic_test_dataset = CIRRDataset('test1', 'classic', preprocess)
    index_features, index_features_p, index_names = extract_index_features(classic_test_dataset, model, blip_stage1=True)
    relative_test_dataset = CIRRDataset('test1', 'relative', preprocess)

    # Generate test prediction dicts for CIRR
    pairid_to_predictions, pairid_to_group_predictions = generate_cirr_test_dicts(relative_test_dataset,
                                                                                  model,
                                                                                  index_features, index_features_p,
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
                             blip_model: torch.nn.Module, 
                             index_features: torch.tensor, index_features_normed_pooled: torch.tensor,
                             index_names: List[str]) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Compute test prediction dicts for CIRR dataset
    :param relative_test_dataset: CIRR test dataset in relative mode
    :param model: stage1 model
    :param index_features: test index features
    :param index_names: test index names
    :return: Top50 global and Top3 subset prediction for each query (reference_name, caption)
    """

    # Generate predictions
    predicted_features, reference_names, group_members, pairs_id = \
        generate_cirr_test_predictions(blip_model, relative_test_dataset, index_names, index_features)

    print(f"Compute CIRR prediction dicts")

    # Normalize the target candidates' index features
    index_features = index_features_normed_pooled.float() # already normed

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names), -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_predictions = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                             zip(pairs_id, sorted_index_names)}
    pairid_to_group_predictions = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                   zip(pairs_id, sorted_group_names)}

    if 'SAVE_TOPK' in globals():
        if SAVE_TOPK:
            breakpoint() # confirm saving to file

            save_dir = os.path.dirname(os.path.dirname(STAGE1_PATH)) + f'/cirr_top_{K_VALUE}_{relative_test_dataset.split}.pt'
            # by default save to the experiment folder
            torch.save({
                        'sorted_index_names': sorted_index_names[:,:K_VALUE],
                        'index_names': index_names,
                        'split': relative_test_dataset.split,
                    }, save_dir)
            print(f"top {K_VALUE} saved at {save_dir}.")
    else:
        pass

    return pairid_to_predictions, pairid_to_group_predictions


def generate_cirr_test_predictions(blip_model: torch.nn.Module,
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
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32,
                                      num_workers=8, pin_memory=True)

    # Get a mapping from index names to index features
    name_to_feat = dict(zip(index_names, index_features))

    # Initialize pairs_id, predicted_features, group_members and reference_names
    pairs_id = []
    predicted_features = torch.empty((0, 256)).to(device, non_blocking=True)
    group_members = []
    reference_names = []

    for batch_pairs_id, batch_reference_names, captions, batch_group_members in tqdm(
            relative_test_loader):  # Load data
        batch_group_members = np.array(batch_group_members).T.tolist()

        # Compute the predicted features
        with torch.no_grad():
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if len(batch_pairs_id) == 1:
                reference_image_features = itemgetter(*batch_reference_names)(name_to_feat).unqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*batch_reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = blip_model.img_txt_fusion(reference_image_features, None, captions, train=False)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        group_members.extend(batch_group_members)
        reference_names.extend(batch_reference_names)
        pairs_id.extend(batch_pairs_id)

    return predicted_features, reference_names, group_members, pairs_id


def main():
    parser = ArgumentParser()
    parser.add_argument("--submission-name", type=str, required=True, help="submission file name")

    parser.add_argument("--stage1-path", type=str, help="path to trained Stage1")

    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['squarepad', 'targetpad'] ")

    parser.add_argument("--save-topk", dest="save_topk", action='store_true',
                        help="Whether to save the topK results")
    parser.add_argument("--k", default=50, type=int,
                        help="top-k value to be saved for test1 split. for CIRR: default 50.")
    args = parser.parse_args()

    args.stage1_path = get_model_path(args.stage1_path, stage1=True)

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


    generate_cirr_test_submissions(model, preprocess, args.submission_name)


if __name__ == '__main__':
    main()
