import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
from torch.autograd import Variable
from Lime import lime_vis
from saliency import sal_map
from smooth_grad import smooth_vis
from Filter import fil_exp
from Integrated_gradient import int_gradient
from argparse import ArgumentParser
from torchsummary import summary
# Model definition
class Classifier(nn.Module):
  def __init__(self):
    super(Classifier, self).__init__()

    def building_block(indim, outdim):
      return [
        nn.Conv2d(indim, outdim, 3, 1, 1),
        nn.BatchNorm2d(outdim),
        nn.ReLU(),
      ]
    def stack_blocks(indim, outdim, block_num):
      layers = building_block(indim, outdim)
      for i in range(block_num - 1):
        layers += building_block(outdim, outdim)
      layers.append(nn.MaxPool2d(2, 2, 0))
      return layers

    cnn_list = []
    cnn_list += stack_blocks(3, 128, 3)
    cnn_list += stack_blocks(128, 128, 3)
    cnn_list += stack_blocks(128, 256, 3)
    cnn_list += stack_blocks(256, 512, 1)
    cnn_list += stack_blocks(512, 512, 1)
    self.cnn = nn.Sequential( * cnn_list)

    dnn_list = [
      nn.Linear(512 * 4 * 4, 1024),
      nn.ReLU(),
      nn.Dropout(p = 0.3),
      nn.Linear(1024, 11),
    ]
    self.fc = nn.Sequential( * dnn_list)

  def forward(self, x):
    out = self.cnn(x)
    out = out.reshape(out.size()[0], -1)
    return self.fc(out)


# Dataset definition
class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        # mode: 'train' or 'eval'
        
        self.paths = paths
        self.labels = labels
        trainTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        evalTransform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),
        ])
        self.transform = trainTransform if mode == 'train' else evalTransform

    # pytorch dataset class
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        X = Image.open(self.paths[index])
        X = self.transform(X)
        Y = self.labels[index]
        return X, Y

    # help to get images for visualizing
    def getbatch(self, indices):
        images = []
        labels = []
        for index in indices:
          image, label = self.__getitem__(index)
          images.append(image)
          labels.append(label)
        return torch.stack(images), torch.tensor(labels)

# help to get data path and label
def get_paths_labels(path):
    def my_key(name):
      return int(name.replace(".jpg",""))+1000000*int(name.split("_")[0])
    imgnames = os.listdir(path)
    imgnames.sort(key=my_key)
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels

if __name__ == '__main__':
    if not os.path.exists('results'):
        os.makedirs('results')

    parser = ArgumentParser()
    parser.add_argument('--ckptpath', type=str, default='./pretrain/checkpoint.pth',
                        help='pretrain model')
    parser.add_argument('--dataset_dir', type=str, default='./food/',
                        help='pretrain model')
    parser.add_argument('--lime', action='store_true', default=False,
                        help='show with lime')
    parser.add_argument('--smooth', action='store_true', default=False,
                        help='show with smooth gradient')
    parser.add_argument('--filter', action='store_true', default=False,
                        help='show with Filter explanation')
    parser.add_argument('--int_grad', action='store_true', default=False,
                        help='show with Integrated Gradient')

    args = parser.parse_args()
    # Load trained model
    model = Classifier().cuda()
    summary(model,(3,128,128))
    checkpoint = torch.load(args.ckptpath)
    model.load_state_dict(checkpoint['model_state_dict'])

    train_paths, train_labels = get_paths_labels(args.dataset_dir)
    train_set = FoodDataset(train_paths, train_labels, mode='eval')

    img_indices = [i for i in range(len(train_set))]
    images, labels = train_set.getbatch(img_indices)
    fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[i].imshow(img.cpu().permute(1, 2, 0))
    plt.savefig('results/org_image.jpg')

    sal_map(images, labels, model,img_indices)

    if args.lime:
        lime_vis(images,labels,model,img_indices)
    if args.smooth:
        smooth_vis(images, labels, model,img_indices)
    if args.filter:
        fil_exp(images, labels, model,img_indices,train_set)
    if args.int_grad:
        int_gradient(images, labels, model,img_indices,train_set)
