import matplotlib.pyplot as plt
import torch
import torch.nn as nn

#####################################################################
# Saliency map 
#####################################################################

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())
  # return torch.log(image)/torch.log(image.max())

def compute_saliency_maps(x, y, model):
  model.eval()
  x = x.cuda()

  # we want the gradient of the input x
  x.requires_grad_()
  
  y_pred = model(x)
  loss_func = torch.nn.CrossEntropyLoss()
  loss = loss_func(y_pred, y.cuda())
  loss.backward()

  # saliencies = x.grad.abs().detach().cpu()
  saliencies, _ = torch.max(x.grad.data.abs().detach().cpu(),dim=1)

  # We need to normalize each image, because their gradients might vary in scale, but we only care about the relation in each image
  saliencies = torch.stack([normalize(item) for item in saliencies])
  return saliencies

def sal_map(images, labels, model,img_indices):
    # images, labels = train_set.getbatch(img_indices)
    saliencies = compute_saliency_maps(images, labels, model)

    # visualize
    fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
    for row, target in enumerate([images, saliencies]):
        for column, img in enumerate(target):
            if row==0:
                axs[row][column].imshow(img.permute(1, 2, 0).numpy())
                # What is permute?
                # In pytorch, the meaning of each dimension of image tensor is (channels, height, width)
                # In matplotlib, the meaning of each dimension of image tensor is (height, width, channels)
                # permute is a tool for permuting dimensions of tensors
                # For example, img.permute(1, 2, 0) means that,
                # - 0 dimension is the 1 dimension of the original tensor, which is height
                # - 1 dimension is the 2 dimension of the original tensor, which is width
                # - 2 dimension is the 0 dimension of the original tensor, which is channels
            else:
                axs[row][column].imshow(img.numpy(), cmap=plt.cm.hot)
    plt.savefig('results/saliency_map.jpg')
            