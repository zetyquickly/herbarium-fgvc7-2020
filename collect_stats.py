from dataset import HerbalDataset
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import pretrainedmodels.utils as utils
import math

# tf_img = utils.TransformImage({
#     'input_space':'RGB', 
#     'input_range':[0,1], 
#     'input_size': [3,768,768],
#     'mean': [0.,0.,0.],
#     'std' : [1.,1.,1.]
# })

input_size = [3,768,768]
scale = 0.875
tf_img = transforms.Compose(
    [
        transforms.Resize(int(math.floor(max(input_size)/scale))),
        transforms.CenterCrop(max(input_size)),
        transforms.ToTensor()
    ]
)

data_train = HerbalDataset('./dataset/nybg2020/train/', train=True, transform=tf_img)
data_test = HerbalDataset('./dataset/nybg2020/test/', train=False, transform=tf_img)

data_whole =  ConcatDataset([data_train, data_test])
batch_size = 24*2
loader = DataLoader(data_whole, batch_size=batch_size, num_workers=24)

device = 'cuda:2'
mean = torch.zeros(3).to(device)
std2 = torch.zeros(3).to(device)
nb_samples = 0

for batch in tqdm(loader):
    data = batch['image'].to(device)
    batch_samples = data.size(0)
    data = data.view(data.size(1), -1)
    
    mu_new = data.mean(1)
    mu_old = mean

    mean = (nb_samples * mean / (nb_samples + batch_samples) + batch_samples * mu_new / 
            (nb_samples + batch_samples)) 

    std2 = (nb_samples * (std2 + mu_old ** 2) / (nb_samples + batch_samples) + 
            batch_samples * (data.std(1) ** 2 + mu_new ** 2) / (nb_samples + batch_samples) - 
            mean ** 2)
    
    nb_samples += batch_samples
    if nb_samples // batch_samples % 300 == 0:
        with open('./stats', 'w') as f:
            f.write(str(mean) + '\n')
            f.write(str(torch.sqrt(std2)) + '\n')
