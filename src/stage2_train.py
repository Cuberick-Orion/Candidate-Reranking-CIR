import os, socket
num_numpy_threads = '8'
os.environ['OPENBLAS_NUM_THREADS'] = num_numpy_threads
os.environ['GOTO_NUM_THREADS'] = num_numpy_threads
os.environ['MKL_NUM_THREADS'] = num_numpy_threads
os.environ['NUMEXPR_NUM_THREADS'] = num_numpy_threads
os.environ['OMP_NUM_THREADS'] = num_numpy_threads

from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from data_utils import base_path, squarepad_transform, FashionIQDataset, targetpad_transform, CIRRDataset
from blip_stage1 import blip_stage1
from blip_stage2 import blip_stage2
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features, generate_randomized_fiq_caption, device
from validate_stage2 import compute_cirr_val_metrics, compute_fiq_val_metrics

def get_model_path(model_path, dataset):
    '''
    helper function to obtain full model path, for the stageI model checkpoint

    Assume the actual checkpoint path to be like:
    (for FashionIQ)  models/<EXP_FOLDER_NAME>/saved_models/blip.pt
    (for CIRR)       models/<EXP_FOLDER_NAME>/saved_models/blip_mean.pt
    for both stageI and stageII.

    You can only provide the <EXP_FOLDER_NAME> string, this function can complete the rest.
    '''
    if model_path is None:
        return None
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


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
    return lr

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):        
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr    
    return lr

def exp_lr_schedule(optimizer, epoch, gamma):
    """Decay the learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma
    return None

def classifier_training_fiq(train: bool,
                          train_dress_types: List[str], val_dress_types: List[str],
                          num_workers: int, num_epochs: int,
                          blip_learning_rate: float, blip_max_epoch:int, blip_img_tune: bool, batch_size: int, blip_bs: int, validation_frequency: int, preprocess_val: bool, 
                          transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    Train the stage II model
    """

    EXPERIMENT_TAG = "CIR_Reranking"
    STAGE_TAG = "stageII"
    experiment_name = kwargs.get("experiment_name")
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/{EXPERIMENT_TAG}_{STAGE_TAG}_fiq_{training_start}_{socket.gethostname()}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"training start time {training_start}")
    print(f"local folder {training_path}")
    print(f"experiment_name {experiment_name}")

    top_k_path = kwargs.get("top_k_path")
    print(f"Top-K pt file loaded at {top_k_path}")
    K = kwargs.get("K_value")
    print(f"Candidate selection: top-{K}")

    # Save all the hyperparameters on a file
    with open(training_path / f"{experiment_name}.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    input_dim = 384

    # load the stageI trained model for extracting z_t embeddings on the fly
    # alternatively, this can be pre-processed offline and saved to a local file (TODO)
    import yaml
    config = yaml.load(open('configs/retrieval_coco.yaml', 'r'), Loader=yaml.Loader)
    model_stage1 = blip_stage1(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    model_stage1 = model_stage1.to(device) 

    if kwargs.get("blip_model_path"):
        print('Trying to load the trained stageI model')
        blip_model_path = kwargs["blip_model_path"]
        state_dict = torch.load(blip_model_path, map_location=device)
        model_stage1.load_state_dict(state_dict["BLIP_Retrieval"])
        print('stageI model loaded successfully')
    else:
        raise RuntimeWarning("not loading trained stageI model?")
        
    for param in model_stage1.parameters():
        param.requires_grad = False

    config = yaml.load(open('configs/nlvr.yaml', 'r'), Loader=yaml.Loader)
    model = blip_stage2(pretrained=config['pretrained'], image_size=config['image_size'], med_config='configs/med_config.json',
                    vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)   

    if not blip_img_tune:
        print('Only the text encoder will be fine-tuned')
        for param in model.visual_encoder.parameters():
            param.requires_grad = False

    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M')

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")

    if not blip_img_tune and preprocess_val:
        model.eval()
        idx_to_dress_mapping = {}
        relative_val_datasets = []
        index_features_list = []
        index_names_list = []

        for idx, dress_type in enumerate(val_dress_types):
            idx_to_dress_mapping[idx] = dress_type
            relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, load_topk=top_k_path.replace('DTYPE',dress_type), K=K)
            relative_val_datasets.append(relative_val_dataset)
            classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, load_topk=top_k_path.replace('DTYPE',dress_type), K=K)
            index_features_and_names = extract_index_features(classic_val_dataset, model, blip_stage2=True)
            index_features_list.append(index_features_and_names[0])
            index_names_list.append(index_features_and_names[1])
            torch.cuda.empty_cache()

    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=num_workers, 
                                       pin_memory=True, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=blip_learning_rate, weight_decay=config['weight_decay'])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    grad_accumulation_step = kwargs.get("grad_accumulation_step")

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        if train:
            relative_train_loader.dataset.epoch_count = epoch
            print(f"[{datetime.now()}] Training ...")
            updated_lr = cosine_lr_schedule(optimizer, epoch, blip_max_epoch, blip_learning_rate, config['min_lr'])
            try: experiment.log_metric('epoch_lr', updated_lr, epoch=epoch)
            except: pass

            with experiment.train():
                train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}

                model.train()
                model_stage1.train()
                train_bar = tqdm(relative_train_loader, ncols=75)
                for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                    step = len(train_bar) * epoch + idx
                    images_in_batch = reference_images.size(0)

                    optimizer.zero_grad()

                    # prepare text
                    # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                    flattened_captions: list = np.array(captions).T.flatten().tolist()
                    input_captions = generate_randomized_fiq_caption(flattened_captions)
                    # prepare target
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    # embed images
                    reference_images = reference_images.to(device, non_blocking=True)
                    target_images = target_images.to(device, non_blocking=True)
                    if not blip_img_tune:
                        with torch.no_grad():
                            reference_images_list = torch.split(reference_images, blip_bs)
                            reference_image_features = torch.vstack(
                                [model.img_embed(mini_batch).float() for mini_batch in reference_images_list])
                            target_images_list = torch.split(target_images, blip_bs)
                            target_image_features = torch.vstack(
                                [model.img_embed(mini_batch).float() for mini_batch in target_images_list])
                    else:
                        with torch.cuda.amp.autocast():
                            reference_images_list = torch.split(reference_images, blip_bs)
                            reference_image_features = torch.vstack(
                                [model.img_embed(mini_batch).float() for mini_batch in reference_images_list])
                            target_images_list = torch.split(target_images, blip_bs)
                            target_image_features = torch.vstack(
                                [model.img_embed(mini_batch).float() for mini_batch in target_images_list])

                    with torch.no_grad():
                        # extract z_t embeddings from stageI model on-the-fly
                        z_t = model_stage1.img_txt_fusion(reference_image_features, reference_image_features, input_captions, 
                                                             train=False, return_raw=True) # last_hidden_state B x L x 768
                        
                    # Compute the logits and the loss
                    with torch.cuda.amp.autocast():
                        logits = model.img_txt_fusion(z_t, target_image_features, input_captions, train=True)
                        ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                        loss_raw = crossentropy_criterion(logits, ground_truth)

                        loss = loss_raw

                    loss /= grad_accumulation_step
                    # Backpropagate and update the weights
                    scaler.scale(loss).backward()
                    if ((idx + 1) % grad_accumulation_step == 0) or (idx + 1 == len(relative_train_loader)):
                        scaler.step(optimizer)
                        scaler.update()

                    experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                    experiment.log_metric('step_loss_raw', loss_raw.detach().cpu().item(), step=step)
                    update_train_running_results(train_running_results, loss, images_in_batch)
                    set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

                train_epoch_loss = float(
                    train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
                experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

                # Training CSV logging
                training_log_frame = pd.concat(
                    [training_log_frame,
                    pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
                training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

                print(f"{training_path}")

        if epoch % validation_frequency == 0:
            print(f"[{datetime.now()}] Validating...")
            torch.cuda.empty_cache()
            with experiment.validate():
                with torch.no_grad():
                    model.eval()
                    model_stage1.eval()

                    recalls_at10 = []
                    recalls_at50 = []

                    if blip_img_tune or (not preprocess_val): # image encoder being updated, refresh features
                        idx_to_dress_mapping = {}
                        relative_val_datasets = []
                        index_features_list = []
                        index_names_list = []

                        for idx, dress_type in enumerate(val_dress_types):
                            idx_to_dress_mapping[idx] = dress_type
                            relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess, load_topk=top_k_path.replace('DTYPE',dress_type), K=K)
                            relative_val_datasets.append(relative_val_dataset)
                            classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess, load_topk=top_k_path.replace('DTYPE',dress_type), K=K)
                            index_features_and_names = extract_index_features(classic_val_dataset, model, blip_stage2=True)
                            index_features_list.append(index_features_and_names[0])
                            index_names_list.append(index_features_and_names[1])
                            torch.cuda.empty_cache()

                    # Compute and log validation metrics for each validation dataset (which corresponds to a different
                    # FashionIQ category)
                    for relative_val_dataset, index_features, index_names, idx in zip(relative_val_datasets,
                                                                                    index_features_list,
                                                                                    index_names_list,
                                                                                    idx_to_dress_mapping):
                        recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, model, model_stage1, index_features,
                                                                        index_names)
                        recalls_at10.append(recall_at10)
                        recalls_at50.append(recall_at50)

                    if not preprocess_val:
                        del index_features_list
                        torch.cuda.empty_cache()

                    results_dict = {}
                    for i in range(len(recalls_at10)):
                        results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                        results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                    results_dict.update({
                        f'average_recall_at10': mean(recalls_at10),
                        f'average_recall_at50': mean(recalls_at50),
                        f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
                    })

                    print(json.dumps(results_dict, indent=4))
                    experiment.log_metrics(
                        results_dict,
                        epoch=epoch
                    )

                    # Validation CSV logging
                    log_dict = {'epoch': epoch}
                    log_dict.update(results_dict)
                    validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                    validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                # Save model
                if save_training and train:
                    save_model('blip_last', epoch, model, training_path, optimizer)
                    if save_best and results_dict['average_recall'] > best_avg_recall:
                        best_avg_recall = results_dict['average_recall']
                        save_model('blip', epoch, model, training_path, optimizer)
                    elif not save_best:
                        save_model(f'blip_{epoch}', epoch, model, training_path, optimizer)

                    print(f"{training_path}/saved_models/blip.pt")

            if not train:
                exit()

def classifier_training_cirr(train: bool,
                           num_workers: int, num_epochs: int,
                           blip_learning_rate: float, blip_max_epoch:int, blip_img_tune: bool, batch_size: int, blip_bs: int, validation_frequency: int,  preprocess_val: bool, 
                           transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    Train the stageII model
    """

    EXPERIMENT_TAG = "CIR_Reranking"
    STAGE_TAG = "stageII"
    experiment_name = kwargs.get("experiment_name")
    training_start = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    training_path: Path = Path(
        base_path / f"models/{EXPERIMENT_TAG}_{STAGE_TAG}_cirr_{training_start}_{socket.gethostname()}")
    training_path.mkdir(exist_ok=False, parents=True)
    print(f"training start time {training_start}")
    print(f"local folder {training_path}")
    print(f"experiment_name {experiment_name}")

    top_k_path = kwargs.get("top_k_path")
    print(f"Top-K pt file loaded at {top_k_path}")
    K = kwargs.get("K_value")
    print(f"Candidate selection: top-{K}")

    # Save all the hyperparameters on a file
    with open(training_path / f"{experiment_name}.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    input_dim = 384

    import yaml
    config = yaml.load(open('configs/retrieval_coco.yaml', 'r'), Loader=yaml.Loader)
    model_stage1 = blip_stage1(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                    vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])
    
    model_stage1 = model_stage1.to(device) 

    if kwargs.get("blip_model_path"):
        print('Trying to load the stageI trained model')
        blip_model_path = kwargs["blip_model_path"]
        state_dict = torch.load(blip_model_path, map_location=device)
        model_stage1.load_state_dict(state_dict["BLIP_Retrieval"])
        print('stageI model loaded successfully')
    else:
        raise RuntimeWarning("not loading pre-trained stageI trained model?")
        
    for param in model_stage1.parameters():
        param.requires_grad = False

    config = yaml.load(open('configs/nlvr.yaml', 'r'), Loader=yaml.Loader)
    model = blip_stage2(pretrained=config['pretrained'], image_size=config['image_size'], med_config='configs/med_config.json',
                    vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'])

    model = model.to(device)   

    if not blip_img_tune:
        print('Only the text encoder will be fine-tuned')
        for param in model.visual_encoder.parameters():
            param.requires_grad = False

    print(f'Number of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M')

    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad": # default
        target_ratio = kwargs['target_ratio'] # 1.25
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['squarepad', 'targetpad']")


    # Define the validation datasets and extract the validation index features
    if not blip_img_tune and preprocess_val:
        model.eval()
        relative_val_dataset = CIRRDataset('val', 'relative', preprocess, load_topk=top_k_path, K=K)
        classic_val_dataset = CIRRDataset('val', 'classic', preprocess, load_topk=top_k_path, K=K)
        val_index_features, val_index_names = extract_index_features(classic_val_dataset, model, blip_stage2=True)
        torch.cuda.empty_cache()

    relative_train_dataset = CIRRDataset('train', 'relative', preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size, 
                                       num_workers=num_workers,
                                       pin_memory=True, collate_fn=collate_fn, 
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=blip_learning_rate, weight_decay=config['weight_decay'])
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    grad_accumulation_step = kwargs.get("grad_accumulation_step")

    # When save_best == True initialize the best results to zero
    if save_best:
        best_mean = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')
    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        if train:
            relative_train_loader.dataset.epoch_count = epoch
            print(f"[{datetime.now()}] Training ...")
            updated_lr = cosine_lr_schedule(optimizer, epoch, blip_max_epoch, blip_learning_rate, config['min_lr'])
            try: experiment.log_metric('epoch_lr', updated_lr, epoch=epoch)
            except: pass
            
            with experiment.train():
                train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}

                model.train()
                model_stage1.train()
                train_bar = tqdm(relative_train_loader, ncols=75)
                for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                    images_in_batch = reference_images.size(0)
                    step = len(train_bar) * epoch + idx

                    optimizer.zero_grad()

                    # prepare target
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    # embed images
                    reference_images = reference_images.to(device, non_blocking=True)
                    target_images = target_images.to(device, non_blocking=True)
                    if not blip_img_tune:
                        with torch.no_grad():
                            reference_images_list = torch.split(reference_images, blip_bs)
                            reference_features = torch.vstack(
                                [model.img_embed(mini_batch).float() for mini_batch in reference_images_list])
                            target_images_list = torch.split(target_images, blip_bs)
                            target_features = torch.vstack(
                                [model.img_embed(mini_batch).float() for mini_batch in target_images_list])
                    else:
                        with torch.cuda.amp.autocast():
                            reference_images_list = torch.split(reference_images, blip_bs)
                            reference_features = torch.vstack(
                                [model.img_embed(mini_batch).float() for mini_batch in reference_images_list])
                            target_images_list = torch.split(target_images, blip_bs)
                            target_features = torch.vstack(
                                [model.img_embed(mini_batch).float() for mini_batch in target_images_list])

                    with torch.no_grad():
                        z_t = model_stage1.img_txt_fusion(reference_features, reference_features, captions, 
                                                             train=False, return_raw=True) # last_hidden_state B x L x 768
                        
                    # Compute the logits and loss
                    with torch.cuda.amp.autocast():
                        logits = model.img_txt_fusion(z_t, target_features, captions, train=True)
                        ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                        loss_raw = crossentropy_criterion(logits, ground_truth)

                        loss = loss_raw

                    loss /= grad_accumulation_step
                    # Backpropagate and update the weights
                    scaler.scale(loss).backward()
                    if ((idx + 1) % grad_accumulation_step == 0) or (idx + 1 == len(relative_train_loader)):
                        scaler.step(optimizer)
                        scaler.update()

                    experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                    experiment.log_metric('step_loss_raw', loss_raw.detach().cpu().item(), step=step)
                    update_train_running_results(train_running_results, loss, images_in_batch)
                    set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

            print(f"{training_path}")

        if epoch % validation_frequency == 0:
            print(f"[{datetime.now()}] Validating...")
            torch.cuda.empty_cache()
            with experiment.validate():
                with torch.no_grad():
                    model.eval()
                    model_stage1.eval()

                    if blip_img_tune or (not preprocess_val):
                        relative_val_dataset = CIRRDataset('val', 'relative', preprocess, load_topk=top_k_path, K=K)
                        classic_val_dataset = CIRRDataset('val', 'classic', preprocess, load_topk=top_k_path, K=K)
                        val_index_features, val_index_names = extract_index_features(classic_val_dataset, model, blip_stage2=True)
                        torch.cuda.empty_cache()

                    # Compute and log validation metrics
                    results = compute_cirr_val_metrics(relative_val_dataset, model, model_stage1, val_index_features,
                                                    val_index_names)
                    group_recall_at1, group_recall_at2, group_recall_at3, recall_at1, recall_at5, recall_at10, recall_at50 = results
                    torch.cuda.empty_cache()

                    if not preprocess_val:
                        del val_index_features
                        torch.cuda.empty_cache()

                    results_dict = {
                        'group_recall_at1': group_recall_at1,
                        'group_recall_at2': group_recall_at2,
                        'group_recall_at3': group_recall_at3,
                        'recall_at1': recall_at1,
                        'recall_at5': recall_at5,
                        'recall_at10': recall_at10,
                        'recall_at50': recall_at50,
                        'mean(R@5+R_s@1)': (group_recall_at1 + recall_at5) / 2,
                        'arithmetic_mean': mean(results),
                        'harmonic_mean': harmonic_mean(results),
                        'geometric_mean': geometric_mean(results)
                    }

                    print(json.dumps(results_dict, indent=4))
                    experiment.log_metrics(
                        results_dict,
                        epoch=epoch
                    )

                    # Validation CSV logging
                    log_dict = {'epoch': epoch}
                    log_dict.update(results_dict)
                    validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                    validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

                # Save model
                if save_training and train:
                    save_model('blip_last', epoch, model, training_path, optimizer)
                    if save_best and results_dict['mean(R@5+R_s@1)'] > best_mean:
                        best_mean = results_dict['mean(R@5+R_s@1)']
                        save_model('blip_mean', epoch, model, training_path, optimizer)
                    if not save_best:
                        save_model(f'blip_{epoch}', epoch, model, training_path, optimizer)

                    print(f"{training_path}/saved_models/blip_mean.pt")

            if not train:
                exit()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train", dest="train", action='store_true', help="whether to do train")
    parser.add_argument("--grad-accumulation-step", default=1, type=int, help="grad accu")
    parser.add_argument("--preprocess-val", dest="preprocess_val", action='store_true', help="whether to preprocess val features, which will occupy around 32G of VRAM on fiq -- we do not advise using this function unless you are sure your VRAM is sufficiently large")

    parser.add_argument("--dataset", type=str, required=True, help="should be either 'CIRR' or 'fashionIQ'")
    parser.add_argument("--num_workers", type=int, required=False, default=4, help="For low memory consumption: limit cirr to 4 and fashioniq to 1 or 2. For optimal training efficiency, set to 8")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")

    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")

    parser.add_argument("--top-k-path", type=str, required=True, help="Path to the top-k pt file")
    parser.add_argument("--K-value", type=int, required=True, help="K value used for validation")

    parser.add_argument("--blip-model-path", type=str, help="Path to the fine-tuned stageI blip model")

    parser.add_argument("--blip-learning-rate", default=2e-5, type=float, help="BLIP learning rate")
    parser.add_argument("--blip-max-epoch", default=20, type=int, help="BLIP max epoch")
    parser.add_argument("--blip-img-tune", dest="blip_img_tune", action='store_true', help="BLIP finetune image encoder, we do not finetune it in this work but preserve this option in code")

    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size of the stageII training")
    parser.add_argument("--blip-bs", default=32, type=int, help="Batch size during BLIP image feature extraction")

    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'cirr']:
        raise ValueError("Dataset should be either 'CIRR' or 'FashionIQ")

    training_hyper_params = {
        "num_epochs": args.num_epochs,
        "top_k_path": get_top_k_path(args.top_k_path, args.dataset.lower()),
        "K_value": args.K_value,
        "blip_model_path": get_model_path(args.blip_model_path, args.dataset.lower()),
        "blip_learning_rate": args.blip_learning_rate,
        "blip_max_epoch": args.blip_max_epoch,
        "blip_img_tune": args.blip_img_tune,
        "batch_size": args.batch_size,
        "blip_bs": args.blip_bs,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
        "num_workers": args.num_workers,
        "experiment_name": args.experiment_name,
        "train": args.train,
        "grad_accumulation_step": args.grad_accumulation_step,
        "preprocess_val": args.preprocess_val,
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"Candidate Rerank {args.dataset}",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    random_seed = 0
    print(f"setting random seed to {random_seed}")
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = True
    import numpy as np
    np.random.seed(random_seed)
    
    if args.dataset.lower() == 'cirr':
        classifier_training_cirr(**training_hyper_params)
    elif args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        classifier_training_fiq(**training_hyper_params)