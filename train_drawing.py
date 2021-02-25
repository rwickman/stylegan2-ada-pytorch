import copy
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import caffe

import dnnlib
import legacy
import cv2
import matplotlib.pyplot as plt
# Steps:
# Sample batch of images from StyleGAN
# Turn images into "drawings"
# Run "drawings" through VGG network to get w vector
# Run w vector though StyleGAN
# Compute loss between original images and images reconstructed from StyleGAN
# Train VGG network

# def load_edge_detect():

#! [CropLayer]
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = (inputShape[2] - targetShape[2]) // 2
        self.xstart = (inputShape[3] - targetShape[3]) // 2
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

cv2.dnn_registerLayer('Crop', CropLayer)
pretrained_model = "models/edge_detection/hed_pretrained_bsds.caffemodel"
model_def = "models/edge_detection/deploy.prototxt"
edge_net = cv2.dnn.readNetFromCaffe(model_def, pretrained_model)


truncation_psi = 0.5
seed = 150
device = torch.device('cuda')
network_pkl = "network-snapshot-000112.pkl"
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore


G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

z = torch.from_numpy(np.random.RandomState().randn(1, G.z_dim)).to(device)
img = G(z, None, truncation_psi=truncation_psi)
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img = img[0].cpu().numpy()
#plt.imshow()

# Load VGG16 feature detector.
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
with dnnlib.util.open_url(url) as f:
    vgg16 = torch.jit.load(f)#.eval().to(device)
optimizer = torch.optim.Adam(vgg16.parameters())

print(vgg16)
# Convert to edges


inp = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(512, 512),
                        mean=(104.00698793, 116.66876762, 122.67891434),
                        swapRB=False, crop=False)


edge_net.setInput(inp)
out = edge_net.forward()
out = out[0, 0]
# Unnormalize output
out = (out * 255).astype(np.uint8)

out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
fig, axs = plt.subplots(1,2)
axs[0].imshow(img)
axs[1].imshow(out, cmap='gray')
axs[0].axis("off")
axs[1].axis("off")
plt.show()

# # Load VGG16 feature detector.
# url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
# with dnnlib.util.open_url(url) as f:
#     vgg16 = torch.jit.load(f).eval().to(device)

# # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
# synth_images = (synth_images + 1) * (255/2)
# if synth_images.shape[2] > 256:
#     synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')