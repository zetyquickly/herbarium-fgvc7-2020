import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms

import os
from collections import defaultdict
from tqdm import tqdm

from pycocotools.coco import COCO
import pretrainedmodels.utils as utils

import numpy as np


class HerbalDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, tta_transform=None, tta_times=1):
        self.data_dir = data_dir
        self.coco = COCO(os.path.join(data_dir, 'metadata_utf8.json'))
        self.load_img = utils.LoadImage()
        self.transform = transform
        self.train = train
        self.tta_times = tta_times
        self.tta_transform = tta_transform
        self._prepare_mappings()

    def __len__(self):
        return len(self.coco.imgs)

    def __getitem__(self, idx):
        img_dict = self.coco.imgs[idx]
        img = self.load_img(os.path.join(self.data_dir, img_dict['file_name']))
        if self.train:
            if self.transform is not None:
                img = self.transform(img)
            output_tensor = img
        else:
            output_tensor = [self.transform(img)]
            if self.tta_transform is not None:
                for _ in range(self.tta_times - 1):
                    proc_img = self.tta_transform(img)
                    output_tensor.append(proc_img)
            output_tensor = torch.stack(output_tensor).squeeze()
        
        if self.train:
            ann = self.coco.anns[img_dict['id']]
            cat_dict = self.coco.cats[ann['category_id']]
            img_dict = self.coco.imgs[ann['image_id']]
            region_dict = self.coco.dataset['regions'][ann['region_id']] 
            return {
                'image': output_tensor, 
                'category_id': ann['category_id'],
                'genus_id': self.cat2genus[ann['category_id']], 
                'family_id': self.cat2family[ann['category_id']],
                'id': img_dict['id'], 
            }
        else:
            return {
                'image': output_tensor, 
                'category_id': -1, 
                'family_id': -1,
                'genus_id': -1, 
                'id': img_dict['id'], 
            }
    
    # def train_val_split(self, val_size=0.2, seed=139):
    #     indices, targets = zip(*[(ann['id'], ann['category_id']) for ann in self.coco.anns.values()])
    #     indices = indices + indices
    #     targets = targets + targets
    #     train_indices, val_indices = train_test_split(indices, test_size=val_size, random_state=seed, stratify=targets)
    #     train_dataset = Subset(self, train_indices)
    #     val_dataset = Subset(self, val_indices)
    #     return train_dataset, val_dataset

    def _prepare_mappings(self):
        self.cat2family = {}
        self.cat2genus = {}
        family2id = {}
        genus2id = {}
        fid = 0
        gid = 0
        for cat in self.coco.cats.values():
            if cat['family'] not in family2id:
                family2id[cat['family']] = fid
                fid += 1
            if cat['genus'] not in genus2id:
                genus2id[cat['genus']] = gid
                gid += 1  
            self.cat2family[cat['id']] = family2id[cat['family']]
            self.cat2genus[cat['id']] = genus2id[cat['genus']]
        self.family_num = fid
        self.genus_num = gid

    def get_alphas(self, beta_cat, beta_genus=0.9, beta_family=0.9):
        cat_num = len(self.coco.cats)
        cat_count = torch.zeros(cat_num, dtype=torch.float32)
        genus_count = torch.zeros(self.genus_num, dtype=torch.float32)
        family_count = torch.zeros(self.family_num, dtype=torch.float32)
        for ann in tqdm(self.coco.anns.values()):
            cat_id = ann['category_id']
            cat_count[cat_id] += 1
            genus_count[self.cat2genus[cat_id]] += 1
            family_count[self.cat2family[cat_id]] += 1
        alpha_cat = (1. - beta_cat) / (1. - beta_cat ** cat_count) 
        alpha_genus = (1. - beta_genus) / (1. - beta_genus ** genus_count)
        alpha_family = (1. - beta_family) / (1. - beta_family ** family_count)
        alphas = [alpha_cat, alpha_genus, alpha_family]
        for alpha in alphas:
            alpha[alpha == float("Inf")] = 0.
        return alphas


# class PairedHerbalDataset(HerbalDataset):
#     def __init__(self, dists_dir, eps=0.0, k=10, *args):
#         super(PairedHerbalDataset, self).__init__(args)
#         self.dists_dir = dists_dir
#         self.eps = eps
#         self.k = k

#     def _sample_other_class(self, idx, item_dists, same_class_images):
#         candidate_imgs = item_dists.argsort() 
#         candidate_fns = []
#         for i in range(200):
#             candidate_img_id = candidate_imgs[i]
#             if (candidate_img_id not in same_class_images): 
#                 candidate_fns.append(candidate_img_id)
#             if len(candidate_fns) == self.k: break 
#         np.random.shuffle(candidate_fns) # randomly pick one from K toughest matches
#         return super().__getitem__(candidate_fns[0])

#     def _sample_same_class(self, idx, item_dists, same_class_images):
#         if len(same_class_images) == 1:
#             return _sample_other_class(self, idx, item_dists, same_class_images)
#         else:
            

#     def __getitem__(self, idx):
#         sample_other = idx % 2
#         idx = idx // 2
#         item = super().__getitem__(idx)
#         item_cat = item['category_id']
#         item_dists = torch.load(os.path.join(self.dists_dir, 'dists', str(idx)+'.pt'))
#         same_class_images = self.coco.getImgIds(catIds=[item_cat])

#         if sample_other:
#             pair_item = self._sample_other_class(idx, item_dists, same_class_images)
#         else:
#             pair_item = self._sample_same_class(idx, item_dists, same_class_images)

#         pair_cat = pair_item['category_id']
#         if item_cat == pair_cat:
#             label = 1. - self.eps
#         else:
#             label = 0. + self.eps
#         return item, other_class_item, label


#     def __len__(self):
#         return 2 * len(self.coco.imgs)
