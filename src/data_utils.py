import json
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import warnings
warnings.simplefilter('ignore', PIL.Image.DecompressionBombWarning)

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import random

base_path = Path(__file__).absolute().parents[1].absolute()

def _convert_image_to_rgb(image):
    return image.convert("RGB")


class SquarePad:
    """
    Square pad the input image with zero padding
    """

    def __init__(self, size: int):
        """
        For having a consistent preprocess pipeline with CLIP we need to have the preprocessing output dimension as
        a parameter
        :param size: preprocessing output dimension
        """
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad the image to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    """
    CLIP-like preprocessing transform on a square padded image
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data.
    The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield tuples made of (image_name, image)
        - In 'relative' mode the dataset yield tuples made of:
            - (reference_image, target_image, image_captions) when split == train
            - (reference_name, target_name, image_captions) when split == val
            - (reference_name, reference_image, image_captions) when split == test
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable, **kwargs):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param dress_types: list of fashionIQ category
        :param mode: dataset mode, should be in ['relative', 'classic']:
            - In 'classic' mode the dataset yield tuples made of (image_name, image)
            - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, image_captions) when split == train
                - (reference_name, target_name, image_captions) when split == val
                - (reference_name, reference_image, image_captions) when split == test
        :param preprocess: function which preprocesses the image
        """
        self.epoch_count = None
        
        self.mode = mode
        self.dress_types = dress_types
        self.split = split
        
        # only used when validating on train-split in __getitem__()
        if 'force_validate' in kwargs:
            assert kwargs.get('force_validate') == True
            self.do_validate = kwargs.get('force_validate')
        else:
            self.do_validate = False

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get triplets made by (reference_image, target_image, a pair of relative captions)
        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")
        
        # Stage II
        if 'load_topk' in kwargs:
            top_k_path, K = kwargs.get("load_topk"), kwargs.get("K")
            tmp_f = torch.load(top_k_path)
            assert K <= tmp_f['sorted_index_names'].shape[-1] # sanity check
            assert tmp_f['dress_types'] == dress_type # sanity check
            assert tmp_f['split'] == split # sanity check

            self.K_sorted_index_names = tmp_f['sorted_index_names'][:,:K]
            self.K_labels = tmp_f['labels'][:,:K].numpy()

            self.K_target_names = tmp_f['target_names']
            self.K_index_names = tmp_f['index_names']

            self.K = K


    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                raw_image_captions = self.triplets[index]['captions']
                image_captions = raw_image_captions
                
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    if self.do_validate is True:
                        # force validation returns
                        target_name = self.triplets[index]['target']
                        return reference_name, target_name, image_captions
                    else:
                        reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.jpg"
                        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                        target_name = self.triplets[index]['target']
                        target_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{target_name}.jpg"
                        target_image = self.preprocess(PIL.Image.open(target_image_path))

                        return reference_image, target_image, image_captions

                elif self.split == 'val':
                    if hasattr(self, 'K'):
                        # Stage II
                        target_name = self.triplets[index]['target']
                        return reference_name, target_name, image_captions, self.K_sorted_index_names[index].tolist(), self.K_labels[index]
                    else:
                        # stage I
                        target_name = self.triplets[index]['target']
                        return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{reference_name}.jpg"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.jpg"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")


class CIRRDataset(Dataset):
    """
       CIRR dataset class which manage CIRR data
       The dataset can be used in 'relative' or 'classic' mode:
           - In 'classic' mode the dataset yield tuples made of (image_name, image)
           - In 'relative' mode the dataset yield tuples made of:
                - (reference_image, target_image, rel_caption) when split == train
                - (reference_name, target_name, rel_caption, group_members) when split == val
                - (pair_id, reference_name, rel_caption, group_members) when split == test1
    """

    def __init__(self, split: str, mode: str, preprocess: callable, use_sub=False, **kwargs):
        """
        :param split: dataset split, should be in ['test', 'train', 'val']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                  - In 'classic' mode the dataset yield tuples made of (image_name, image)
                  - In 'relative' mode the dataset yield tuples made of:
                        - (reference_image, target_image, rel_caption) when split == train
                        - (reference_name, target_name, rel_caption, group_members) when split == val
                        - (pair_id, reference_name, rel_caption, group_members) when split == test1
        :param preprocess: function which preprocesses the image
        """
        self.epoch_count = None

        self.preprocess = preprocess
        self.mode = mode
        self.split = split

        # only used when validating on train-split in __getitem__()
        if 'force_validate' in kwargs:
            assert kwargs.get('force_validate') == True
            self.do_validate = kwargs.get('force_validate')
        else:
            self.do_validate = False

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

        # Stage II
        if 'load_topk' in kwargs:
            top_k_path, K = kwargs.get("load_topk"), kwargs.get("K")
            tmp_f = torch.load(top_k_path)
            assert K <= tmp_f['sorted_index_names'].shape[-1] # sanity check
            assert tmp_f['split'] == split # sanity check

            self.K_sorted_index_names = tmp_f['sorted_index_names'][:,:K]
            self.K_index_names = tmp_f['index_names']; assert self.K_index_names == list(self.name_to_relpath.keys()), RuntimeError("Something is wrong.")

            if self.split != 'test1':
                self.K_labels = tmp_f['labels'][:,:K].numpy()
                self.K_group_labels = tmp_f['group_labels'].numpy()

                self.K_target_names = tmp_f['target_names']; assert self.K_target_names == [ii['target_hard'] for ii in self.triplets], RuntimeError("Something is wrong.")

            self.K = K


    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                raw_rel_caption = self.triplets[index]['caption']

                rel_caption = raw_rel_caption


                if self.split == 'train':
                    if self.do_validate is True:
                        # force validation returns
                        target_hard_name = self.triplets[index]['target_hard']
                        return reference_name, target_hard_name, rel_caption, group_members
                    else:
                        reference_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[reference_name]
                        reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                        target_hard_name = self.triplets[index]['target_hard']
                        target_image_path = base_path / 'cirr_dataset' / self.name_to_relpath[target_hard_name]
                        target_image = self.preprocess(PIL.Image.open(target_image_path))

                        return reference_image, target_image, rel_caption

                elif self.split == 'val':
                    if hasattr(self, 'K'):
                        # stage II
                        target_hard_name = self.triplets[index]['target_hard']
                        return reference_name, target_hard_name, rel_caption, group_members, self.K_sorted_index_names[index].tolist(), self.K_labels[index], self.K_group_labels[index]
                    else:
                        # stage I
                        target_hard_name = self.triplets[index]['target_hard']
                        return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    if hasattr(self, 'K'):
                        # stage II
                        pair_id = self.triplets[index]['pairid']
                        return pair_id, reference_name, rel_caption, group_members, self.K_sorted_index_names[index].tolist()
                    else:
                        # stage I
                        pair_id = self.triplets[index]['pairid']
                        return pair_id, reference_name, rel_caption, group_members
                        
            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
