import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
# Smooth grad

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def smooth_grad(x, y, model, epoch, param_sigma_multiplier):
    model.eval()
    #x = x.cuda().unsqueeze(0)

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
    smooth = np.zeros(x.cuda().unsqueeze(0).size())
    for i in range(epoch):
        # call Variable to generate random noise
        noise = Variable(x.data.new(x.size()).normal_(mean, sigma**2))
        x_mod = (x+noise).unsqueeze(0).cuda()
        x_mod.requires_grad_()

        y_pred = model(x_mod)
        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(y_pred, y.cuda().unsqueeze(0))
        loss.backward()

        # like the method in saliency map
        smooth += x_mod.grad.abs().detach().cpu().data.numpy()
    smooth = normalize(smooth / epoch) # don't forget to normalize
    # smooth = smooth / epoch
    return smooth

def smooth_vis(images, labels, model,img_indices):
    # images, labels = train_set.getbatch(img_indices)
    smooth = []
    for i, l in zip(images, labels):
        smooth.append(smooth_grad(i, l, model, 500, 0.4))
    smooth = np.stack(smooth)
    print(smooth.shape)

    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, smooth]):
        for column, img in enumerate(target):
            axs[row][column].imshow(np.transpose(img.reshape(3,128,128), (1,2,0)))
    plt.savefig('results/smooth_grad.jpg')