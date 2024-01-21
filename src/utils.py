import multiprocessing
import random
from pathlib import Path
from typing import Union, Tuple, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import time
from rich import print

from data_utils import CIRRDataset, FashionIQDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def extract_index_features(dataset: Union[CIRRDataset, FashionIQDataset], blip_model,
                           blip_stage2=False, blip_stage1=False) -> \
        Tuple[torch.tensor, List[str]]:
    """
    Extract FashionIQ or CIRR index features
    :param dataset: FashionIQ or CIRR dataset in 'classic' mode
    :return: a tensor of features and a list of images
    """
    feature_dim = 256
    classic_val_loader = DataLoader(dataset=dataset, batch_size=16, num_workers=8,
                                    pin_memory=True, collate_fn=collate_fn)
    index_features = torch.empty((0, feature_dim)).to(device, non_blocking=True)
    index_names = []
    if isinstance(dataset, CIRRDataset):
        print(f"[{datetime.now()}] extracting CIRR {dataset.split} index features")
    elif isinstance(dataset, FashionIQDataset):
        print(f"[{datetime.now()}] extracting fashionIQ {dataset.dress_types} - {dataset.split} index features")
    
    if blip_stage2:
        assert not blip_stage1, ValueError("only one condition shall be selected")
        
        index_features = torch.empty((len(classic_val_loader.dataset), 577, 768)).to(device, non_blocking=True)
        idx_count_ = 0
        for names, images in tqdm(classic_val_loader):
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                batch_features = blip_model.img_embed(images, )
                index_features[idx_count_:idx_count_+batch_features.shape[0]] = batch_features
                idx_count_ += batch_features.shape[0]
                index_names.extend(names)
        return index_features, index_names
    elif blip_stage1:
        assert not blip_stage2, ValueError("only one condition shall be selected")

        index_features = torch.empty((len(classic_val_loader.dataset), 577, 768)).to(device, non_blocking=True)
        index_features_p = torch.empty((len(classic_val_loader.dataset), 256)).to(device, non_blocking=True) # pooled and normalized
        idx_count_ = 0
        for names, images in tqdm(classic_val_loader):
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                batch_features = blip_model.img_embed(images, return_pool_and_normalized=True)
                index_features[idx_count_:idx_count_+batch_features[0].shape[0]] = batch_features[0]
                index_features_p[idx_count_:idx_count_+batch_features[1].shape[0]] = batch_features[1]
                idx_count_ += batch_features[0].shape[0]
                index_names.extend(names)
        return index_features, index_features_p, index_names
    else:
        raise RuntimeError


def generate_randomized_fiq_caption(flattened_captions: List[str]) -> List[str]:
    """
    Function which randomize the FashionIQ training captions in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1
    (d) cap2
    :param flattened_captions: the list of caption to randomize, note that the length of such list is 2*batch_size since
     to each triplet are associated two captions
    :return: the randomized caption list (with length = batch_size)
    """
    captions = []
    for i in range(0, len(flattened_captions), 2):
        random_num = random.random()
        if random_num < 0.25:
            captions.append(
                f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}")
        elif 0.25 < random_num < 0.5:
            captions.append(
                f"{flattened_captions[i + 1].strip('.?, ').capitalize()} and {flattened_captions[i].strip('.?, ')}")
        elif 0.5 < random_num < 0.75:
            captions.append(f"{flattened_captions[i].strip('.?, ').capitalize()}")
        else:
            captions.append(f"{flattened_captions[i + 1].strip('.?, ').capitalize()}")
    return captions


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def update_train_running_results(train_running_results: dict, loss: torch.tensor, images_in_batch: int):
    """
    Update `train_running_results` dict during training
    :param train_running_results: logging training dict
    :param loss: computed loss for batch
    :param images_in_batch: num images in the batch
    """
    train_running_results['accumulated_train_loss'] += loss.to('cpu',
                                                               non_blocking=True).detach().item() * images_in_batch
    train_running_results["images_in_epoch"] += images_in_batch


def set_train_bar_description(train_bar, epoch: int, num_epochs: int, train_running_results: dict):
    """
    Update tqdm train bar during training
    :param train_bar: tqdm training bar
    :param epoch: current epoch
    :param num_epochs: numbers of epochs
    :param train_running_results: logging training dict
    """
    train_bar.set_description(
        desc=f"[{epoch}/{num_epochs}] "
             f"train loss: {train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch']:.3f} "
    )


def save_model(name: str, cur_epoch: int, model_to_save: nn.Module, training_path: Path, optimizer):
    """
    Save the weights of the model during training
    :param name: name of the file
    :param cur_epoch: current epoch
    :param model_to_save: pytorch model to be saved
    :param training_path: path associated with the training run
    """
    models_path = training_path / "saved_models"
    models_path.mkdir(exist_ok=True, parents=True)
    model_name = model_to_save.__class__.__name__
    torch.save({
        'epoch': cur_epoch,
        model_name: model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, str(models_path / f'{name}.pt'))
