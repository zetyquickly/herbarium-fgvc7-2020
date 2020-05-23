import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets import CIFAR10
from torchvision import transforms


import pytorch_lightning as pl
from pytorch_lightning import loggers

import pretrainedmodels
from utils import TransformImage
from pretrainedmodels.utils import Identity

import os
from argparse import Namespace

from dataset import HerbalDataset
from losses import FocalLoss

from albumentations import (
    RandomResizedCrop, HorizontalFlip, RGBShift, Compose
)

from fastai.layers import AdaptiveConcatPool2d

NUM_CATEGORIES = 32094
NUM_GENERA = 3678
NUM_FAMILIES = 310

# class AdamEva(torch.optim.Optimizer):
#     def __init__(self, adam, eva):
#         self.adam = adam
#         self.eva = eva
#         self.param_groups = [self.adam.param_groups, self.eva.param_groups]

#     def zero_grad(self, who):
#         if who == 'adam':
#             self.adam.zero_grad()
#         elif who == 'eva':
#             self.eva.zero_grad()

#     def step(self, who):
#         if who == 'adam':
#             self.adam.step()
#         elif who == 'eva':
#             self.eva.step()

#     def __getitem__(self, who):
#         if who == 'adam':
#             return self.adam
#         elif who == 'eva':
#             return self.eva

#     def __len__(self):
#         return 1

class Head(nn.Module):
    def __init__(self, in_features, NUM_CATEGORIES, NUM_GENERA, NUM_FAMILIES):
        super(Head, self).__init__()
        self.cat_pool = AdaptiveConcatPool2d()
        self.flat = nn.Flatten()
        # self.bn1 = nn.BatchNorm1d(
        #     num_features=2*in_features, eps=1e-05, momentum=0.1, 
        #     affine=True, track_running_stats=True
        # )
        # self.dp1 = nn.Dropout(p=0.5)
        self.linear = nn.Linear(
            in_features=2*in_features, 
            out_features=in_features, bias=True
        )
        self.relu = nn.ReLU()
        # self.bn2 = nn.BatchNorm1d(
        #     num_features=in_features, eps=1e-05, momentum=0.1,
        #     affine=True, track_running_stats=True
        # )
        # self.dp2 = nn.Dropout(p=0.5)
        self.categories_layer = nn.Linear(
            in_features, NUM_CATEGORIES
        )
        self.genera_layer = nn.Linear(
            in_features, NUM_GENERA
        )
        self.families_level = nn.Linear(
            in_features, NUM_FAMILIES
        )

    def pool_flatten(self, x):
        return self.flat(self.cat_pool(x))

    def _base_forward(self, x):
        x = self.pool_flatten(x)
        x = self.relu(self.linear(x))
        # x = self.relu(self.linear(self.dp1(x)))
        # x = self.dp2(self.bn2(x))
        return x

    def forward(self, x):
        x = self._base_forward(x)
        cat_logits = self.categories_layer(x)
        genus_logits = self.genera_layer(x)
        family_logits = self.families_level(x)
        return cat_logits, genus_logits, family_logits

    def inference(self, x):
        x = self._base_forward(x)
        cat_logits = self.categories_layer(x)
        return cat_logits


class Net(pl.LightningModule):
    def __init__(self, hparams):
        super(Net, self).__init__()
        self.hparams = hparams
        self.submodule = pretrainedmodels.se_resnext50_32x4d(
            num_classes=1000, pretrained='imagenet'
        )
        in_features = self.submodule.last_linear.in_features
        # Change last_linear to Identity() to preserve memory
        self.submodule.last_linear = Identity()
        self.head = Head(in_features, NUM_CATEGORIES, NUM_GENERA, NUM_FAMILIES)

    def forward(self, x):
        x = self.submodule.features(x)
        cat_logits, genus_logits, family_logits = self.head(x)
        return cat_logits, genus_logits, family_logits
    
    def inference(self, x):
        x = self.submodule.features(x)
        cat_logits = self.head.inference(x)
        return cat_logits
    
    def embedding(self, x):
        x = self.submodule.features(x)
        return self.head.pool_flatten(x)
    
    def cross_entropy_loss(self, logits, labels):
        logits = torch.log_softmax(logits, dim=1)
        return F.nll_loss(logits, labels)
    
    def focal_loss(self, logits, labels):
        loss_cat = self.focal_loss_cat(logits[0], labels[0])
        loss_genus = self.focal_loss_genus(logits[1], labels[1])
        loss_family = self.focal_loss_family(logits[2], labels[2])
        return 1.5*loss_cat + loss_genus + loss_family

    def training_step(self, train_batch, batch_idx):
        x, y_cat = train_batch['image'], train_batch['category_id']
        y_genus, y_family = train_batch['genus_id'], train_batch['family_id'] 
        logits = self.forward(x)
        y = (y_cat, y_genus, y_family)
        loss = self.focal_loss(logits, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        x, y_cat = val_batch['image'], val_batch['category_id']
        y_genus, y_family = val_batch['genus_id'], val_batch['family_id'] 
        logits = self.forward(x)
        y = (y_cat, y_genus, y_family)
        loss = self.focal_loss(logits, y)
        return {'val_loss': loss}

    def test_step(self, test_batch, batch_idx):
        x, ids = test_batch['image'], test_batch['id']
        batch_size, n_crops, c, h, w = x.size()
        x = x.view(-1, c, h, w) 
        logits = self.inference(x)
        logits = logits.view(batch_size, n_crops, -1).mean(axis=1)
        preds = F.softmax(logits, dim=1)
        return {'batch_preds': preds, 'batch_ids': ids}

    def test_step_end(self, outputs):
        preds = outputs['batch_preds'].cpu()
        ids = outputs['batch_ids'].cpu()
        return {'batch_preds': preds, 'batch_ids': ids}

    def test_epoch_end(self, outputs):
        with open(os.path.join(self.hparams.submission_path, 'submission.csv'), 'w') as f:
            f.write('Id,Predicted\n')
            for x in outputs:
                preds = x['batch_preds']
                _, labels = torch.max(preds, axis=1)
                ids = x['batch_ids']
                for pair in zip(ids, labels):
                    f.write(str(pair[0].item())+','+str(pair[1].item())+'\n')
        return {}
        
    def validation_epoch_end(self, outputs): 
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def prepare_data(self):

        tf_img = TransformImage(
            {
                'input_space':'RGB', 
                'input_range':[0,1], 
                'input_size': [3,512,512],
                'mean': [0.8226, 0.8227, 0.8226],
                'std' : [0.2099, 0.2098, 0.2100],
            }, 
            random_crop=True,
            random_hflip=True, random_vflip=True,
            crop_size=224
        )

        tta_times = 4
        tta_transform = tf_img
        orig_tf = TransformImage(
            {
                'input_space':'RGB', 
                'input_range':[0,1], 
                'input_size': [3,224,224],
                'mean': [0.8226, 0.8227, 0.8226],
                'std' : [0.2099, 0.2098, 0.2100],
            }, 
            random_crop=False,
            random_hflip=False, random_vflip=False,
            crop_size=224
        )

        data_train = HerbalDataset('./dataset/nybg2020/train/', train=True, transform=tf_img)
        self.data_test = HerbalDataset('./dataset/nybg2020/test/', train=False, transform=orig_tf, 
                                        tta_transform=tta_transform, tta_times=tta_times)

        alphas = data_train.get_alphas(beta_cat=0.9, beta_genus=0.9, beta_family=0.9)
        # alphas = [None, None, None]

        self.focal_loss_cat = FocalLoss(gamma=2., alpha=alphas[0])
        self.focal_loss_genus = FocalLoss(gamma=2., alpha=alphas[1])
        self.focal_loss_family = FocalLoss(gamma=2., alpha=alphas[2])

        torch.manual_seed(139)
        val_size = int(0.2 * len(data_train))
        self.data_train, self.data_val = random_split(
            data_train, 
            [len(data_train) - val_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers=24, batch_size=24*7, pin_memory = True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers=24, batch_size=24*7, pin_memory = True)

    def test_dataloader(self):
        return DataLoader(self.data_test, num_workers=24, batch_size=24*3)

    def configure_optimizers(self):
        adam = torch.optim.Adam(self.parameters(), lr=30e-5)
        eva = torch.optim.SGD(self.parameters(), lr=30e-5, momentum=0.8)
        # optimizer = AdamEva(adam, eva)
        return adam

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # if current_epoch % 10 < 5:
        #     who = 'adam'
        # else:
        #     who = 'eva'
        lr = 30e-5
        if current_epoch > 10:
            lr = 15e-5
        if current_epoch > 20:
            lr = 7.5e-5
        if current_epoch > 25:
            lr = 3e-5
        if current_epoch > 30:
            lr = 1e-5
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":

    hparams = Namespace()
    model = Net(hparams)

    # tb_logger = loggers.TensorBoardLogger('logs', name="se_resnext50_32x4d")
    tb_logger = loggers.TensorBoardLogger('logs', name="se_resnext50_32x4d")
    trainer = pl.Trainer(
        gpus=[0,1,2], 
        distributed_backend='dp',
        logger=tb_logger,
        # resume_from_checkpoint='/root/Herbal2020/logs/se_resnext50_32x4d/version_12/checkpoints/epoch=6.ckpt',
        # amp_level='O1', precision=16,
    )
    trainer.fit(model)