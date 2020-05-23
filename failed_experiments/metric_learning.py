import torch
from torch.utils.data import DataLoader, random_split
from dataset import HerbalDataset
from utils import TransformImage
from submissions.submission17.model import Net
import pickle
import os

from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)

def create_similarity_dict(model, dataloader, dir_path='./data/sim_dict_ver17/'):
    model.eval()
    save_batch = dataloader.batch_size
    descs_count = len(dataloader.dataset)
    # idx = 0
    # with torch.no_grad():
    #     for batch in tqdm(dataloader):
    #         ims = batch['image'].cuda()
    #         cnn_out = model.embedding(ims)
    #         batch_descs = cnn_out.detach().cpu()
    #         torch.save(batch_descs, os.path.join(dir_path, 'descs', str(idx+save_batch)+'.pt'))
    #         idx += save_batch

    for batch_num in tqdm(range(0, descs_count, save_batch)):
        cur_desc = torch.load(os.path.join(dir_path, 'descs', str(batch_num + save_batch)+'.pt')).cuda()
        cur_dists = []
        for idx in range(0, descs_count, save_batch):
            batch_descs = torch.load(os.path.join(dir_path, 'descs', str(idx+save_batch)+'.pt')).cuda()
            cur_dists.append(torch.cdist(cur_desc, batch_descs).cpu())
        cur_dists = torch.cat(cur_dists, axis=1)
        torch.save(cur_dists, os.path.join(dir_path, 'dists', str(batch_num + save_batch)+'.pt'))

def save_similarity_dict():
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

    data_train = HerbalDataset('./dataset/nybg2020/train/', train=True, transform=orig_tf)
    dataloader = DataLoader(data_train, num_workers=25, batch_size=500, pin_memory = True)

    PATH = '/root/Herbal2020/submissions/submission17/'

    model = Net.load_from_checkpoint(PATH + 'checkpoints/epoch=28.ckpt')
    model.freeze()
    model.cuda()

    create_similarity_dict(model, dataloader)

    return

if __name__ == "__main__":
    save_similarity_dict()
    # orig_tf = TransformImage(
    #     {
    #         'input_space':'RGB', 
    #         'input_range':[0,1], 
    #         'input_size': [3,224,224],
    #         'mean': [0.8226, 0.8227, 0.8226],
    #         'std' : [0.2099, 0.2098, 0.2100],
    #     }, 
    #     random_crop=False,
    #     random_hflip=False, random_vflip=False,
    #     crop_size=224
    # )

    # data_train = HerbalDataset('./dataset/nybg2020/train/', train=True, transform=orig_tf)
    # PATH = '/root/Herbal2020/submissions/submission17/'

    # model = Net.load_from_checkpoint(PATH + 'checkpoints/epoch=28.ckpt')
    # model.freeze()
    # for idx in tqdm(range(164000, 167000)):
    #     img = data_train[idx]['image'].unsqueeze(0).cuda()
    #     model.embedding(img)
 
