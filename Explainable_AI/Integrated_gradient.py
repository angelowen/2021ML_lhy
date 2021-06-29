import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
class IntegratedGradients():
        def __init__(self, model):
                self.model = model
                self.gradients = None
                # Put model in evaluation mode
                self.model.eval()

        def generate_images_on_linear_path(self, input_image, steps):
                # Generate scaled xbar images
                xbar_list = [input_image*step/steps for step in range(steps)]
                return xbar_list

        def generate_gradients(self, input_image, target_class):
                # We want to get the gradients of the input image
                input_image.requires_grad=True
                # Forward
                model_output = self.model(input_image)
                # Zero grads
                self.model.zero_grad()
                # Target for backprop
                one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
                one_hot_output[0][target_class] = 1
                # Backward
                model_output.backward(gradient=one_hot_output)
                self.gradients = input_image.grad
                # Convert Pytorch variable to numpy array
                # [0] to get rid of the first channel (1,3,128,128)
                gradients_as_arr = self.gradients.data.cpu().numpy()[0]
                return gradients_as_arr

        def generate_integrated_gradients(self, input_image, target_class, steps):
                # Generate xbar images
                xbar_list = self.generate_images_on_linear_path(input_image, steps)
                # Initialize an iamge composed of zeros
                integrated_grads = np.zeros(input_image.size())
                for xbar_image in xbar_list:
                        # Generate gradients from xbar images
                        single_integrated_grad = self.generate_gradients(xbar_image, target_class)
                        # Add rescaled grads from xbar images
                        integrated_grads = integrated_grads + single_integrated_grad/steps
                # [0] to get rid of the first channel (1,3,128,128)
                return integrated_grads[0]

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def int_gradient(images, labels, model,img_indices,train_set):
        # put the image to cuda
        images, labels = train_set.getbatch(img_indices)
        images = images.cuda()
        IG = IntegratedGradients(model)
        integrated_grads = []
        for i, img in enumerate(images):
            img = img.unsqueeze(0)
            integrated_grads.append(IG.generate_integrated_gradients(img, labels[i], 10))
        fig, axs = plt.subplots(2, len(img_indices), figsize=(15, 8))
        for i, img in enumerate(images):
            axs[0][i].imshow(img.cpu().permute(1, 2, 0))
        for i, img in enumerate(integrated_grads):
            axs[1][i].imshow(np.moveaxis(normalize(img),0,-1))
        plt.show()
        plt.savefig('results/Integrated_grad.jpg')
        plt.close()