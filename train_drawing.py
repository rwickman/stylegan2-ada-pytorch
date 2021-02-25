import copy
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
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
def save(vgg16):
    torch.save(vgg16.state_dict(), "models/vgg_net")
def load(vgg16):
    vgg16.load_state_dict(torch.load("models/vgg_net"))


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


truncation_psi = 0.4
seed = 150
device = torch.device('cuda')
network_pkl = "network-snapshot-000712.pkl"
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore


G = copy.deepcopy(G).eval().requires_grad_(False).to(device)


#z = torch.from_numpy(np.random.RandomState().randn(1, G.z_dim)).to(device)
#img = G(z, None, truncation_psi=truncation_psi)
#img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)


# Load VGG16 feature detector.
vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)

# Replace last layer
vgg16.classifier[-1] = nn.Linear(4096, 512)
vgg16.classifier
load(vgg16)
vgg16 = vgg16.train().to(device)

#optimizer = torch.optim.Adam(vgg16.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(vgg16.parameters(), lr=1e-2)
#optimizer = torch.optim.RMSprop(vgg16.parameters(), lr=5e-3)
loss_fn = nn.SmoothL1Loss()

# z = torch.from_numpy(np.random.RandomState(5).randn(1, G.z_dim)).to(device)
# img = G(z, None, noise_mode="const", truncation_psi=truncation_psi)    

total_loss = None
batch_size = 8
#seeds = list(range(64))#[0,1,2,3]

for i in range(256):
    z = torch.from_numpy(np.random.RandomState().randn(1, G.z_dim)).to(device)
    img = G(z, None, noise_mode="none", truncation_psi=truncation_psi)
    # if (i+1) % len(seeds) == 0:
    #     random.shuffle(seeds)
    #img = img.to(device)


    img_temp = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img_temp = img_temp[0].cpu().numpy()
    inp = cv2.dnn.blobFromImage(img_temp, scalefactor=1.0, size=(224, 224),
                        mean=(104.00698793, 116.66876762, 122.67891434),
                        swapRB=False, crop=False)
    edge_net.setInput(inp)
    
    out = edge_net.forward()
    out = out[0, 0]
    out_edge = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    # Invert the colors
    out_edge = 1 - out_edge
    out = torch.tensor(out_edge.T).to(device)
    if img.shape[2] > 256:
        temp_img = F.interpolate(img, size=(224, 224), mode='area')
    temp_img = torch.cat((out.unsqueeze(0), temp_img))
    #print("temp_img.shape", temp_img.shape)
    synth_features = vgg16(temp_img)
    synth_features = synth_features.unsqueeze(1).repeat([1, G.num_ws, 1])
    # Run through StyleGAN network
    # Duplicate w layer
    
    #synth_features = torch.repeat_interleave(synth_features.unsqueeze(1), 16, dim=1)

    img_out = G.synthesis(synth_features, noise_mode="none")#, truncation_psi=truncation_psi)
    w_out = G.mapping(z, None, truncation_psi=truncation_psi)

    

    img_loss = loss_fn(img_out, torch.repeat_interleave(img, 2, dim=0))
    #print("IMG LOSS: ", img_loss)
    w_loss = loss_fn(synth_features[:, 0], torch.repeat_interleave(w_out[:, 0], 2, dim=0))
    # print("W LOSS: ", w_loss)

    # Gradient accumulation
    loss = ((img_loss * 10 + w_loss) / batch_size)
    loss.backward()
    if total_loss is None:
        #print("Setting total loss")
        total_loss = loss
    else:
        #print("Adding to total loss")
        total_loss += loss

    if (i+1) % batch_size == 0:
        optimizer.step()
        print("\nTotal loss: ", total_loss, "\n")
        total_loss = None
        optimizer.zero_grad()
        #vgg16.zero_grad
    
    #loss = loss_fn(img_out, torch.repeat_interleave(img, 2, dim=0)) + 0.01 * 
    #loss = loss_fn(synth_features[:, 0], torch.repeat_interleave(w_out[:, 0], 2, dim=0))
    
    #loss = loss_fn(img_out, torch.repeat_interleave(img, 2, dim=0))
    # print("img_out.shape", img_out.shape)
    # print("img.shape", img.shape)
    #loss = loss_fn(img_out, img)# + loss_fn(synth_features[:, 0], w_out[:, 0])
    
    #print("w_out.shape", w_out[:, 0].shape)

    # print("w[:,0]", w_out[:,0,0].detach().cpu().numpy())
    # print("synth_features[0,0]", synth_features[:,0,0].detach().cpu().numpy())

    # optimizer.zero_grad()
    # loss = img_loss + w_loss
    # loss.backward()
    # optimizer.step()save(vgg16)

#plt.imshow()

# import sys
# sys.exit()

# Convert to edges

# img_out = (img_out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
# img_out = img_out[0].cpu().numpy()


img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img = img[0].cpu().numpy()
inp = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(224, 224),
                        mean=(104.00698793, 116.66876762, 122.67891434),
                        swapRB=False, crop=False)


edge_net.setInput(inp)
out = edge_net.forward()
out = out[0, 0]
# Unnormalize output
out_edge = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

out_edge = 1 - out_edge
out = torch.tensor(out_edge.T).to(device)


# if img.shape[1] > 256:
#     temp_img = F.interpolate(img, size=(224, 224), mode='area')
#synth_features = vgg16(out.unsqueeze(0))
#out = out.permute(1,2,0)


synth_features = vgg16(temp_img)
synth_features = synth_features.unsqueeze(1).repeat([1, G.num_ws, 1])

img_out = G.synthesis(synth_features, noise_mode="const")
img_out = (img_out.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
img_out = img_out.cpu().numpy()

out_edge = (out_edge * 255).astype(np.uint8)

fig, axs = plt.subplots(1,4)
axs[0].imshow(img)
axs[1].imshow(img_out[0])
axs[2].imshow(img_out[1])
axs[3].imshow(out_edge, cmap='gray')

for i in range(4):
    axs[i].axis("off")
plt.show()

save(vgg16)