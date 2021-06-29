from torch.optim import Adam
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

layer_activations = None

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def filter_explanation(x, model, cnnid, filterid, iteration=100, lr=1):
    # x: input image
    # cnnid, filterid: cnn layer id, which filter
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output
    
    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    # When the model forward through the layer[cnnid], need to call the hook function first
    # The hook function save the output of the layer[cnnid]
    # After forwarding, we'll have the loss and the layer activation

    # Filter activation: x passing the filter will generate the activation map
    model(x.cuda()) # forward

    # Based on the filterid given by the function argument, pick up the specific filter's activation map
    # We just need to plot it, so we can detach from graph and save as cpu tensor
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()
    
    # Filter visualization: find the image that can activate the filter the most
    x = x.cuda()
    x.requires_grad_()
    # input image gradient
    optimizer = Adam([x], lr=lr)
    # Use optimizer to modify the input image to amplify filter activation
    for iter in range(iteration):
        optimizer.zero_grad()
        model(x)
        
        objective = -layer_activations[:, filterid, :, :].sum()
        # We want to maximize the filter activation's summation
        # So we add a negative sign
        
        objective.backward()
        # Calculate the partial differential value of filter activation to input image
        optimizer.step()
        # Modify input image to maximize filter activation
    filter_visualizations = x.detach().cpu().squeeze()

    # Don't forget to remove the hook
    hook_handle.remove()
    # The hook will exist after the model register it, so you have to remove it after used
    # Just register a new hook if you want to use it

    return filter_activations, filter_visualizations

def fil_exp(images, labels, model,img_indices,train_set):
    images, labels = train_set.getbatch(img_indices)
    filter_activations, filter_visualizations = filter_explanation(images, model, cnnid=6, filterid=0, iteration=100, lr=0.1)


    fig, axs = plt.subplots(3, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[0][i].imshow(img.permute(1, 2, 0))
    # Plot filter activations
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    # Plot filter visualization
    for i, img in enumerate(filter_visualizations):
        axs[2][i].imshow(normalize(img.permute(1, 2, 0)))
    plt.show()
    plt.savefig('results/filter_explain_1.jpg')
    plt.close()
    
    # activate 的區域對應到一些物品的邊界，尤其是顏色對比較深的邊界

    images, labels = train_set.getbatch(img_indices)
    filter_activations, filter_visualizations = filter_explanation(images, model, cnnid=23, filterid=0, iteration=100, lr=0.1)

    # Plot filter activations
    fig, axs = plt.subplots(3, len(img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[0][i].imshow(img.permute(1, 2, 0))
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    for i, img in enumerate(filter_visualizations):
        axs[2][i].imshow(normalize(img.permute(1, 2, 0)))
    plt.show()
    plt.savefig('results/filter_explain_2.jpg')
    plt.close()
    
