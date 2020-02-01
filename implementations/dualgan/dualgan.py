import argparse
import os
import numpy as np
import math
import itertools
import scipy
import sys
import time
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from datasets import *
from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--load_model", type=str, default='', help="model to load (format: epoch_batch)")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss function
cycle_loss = torch.nn.L1Loss()

# Loss weights
lambda_adv = 1
lambda_cycle = 10
lambda_gp = 10

# Initialize generator and discriminator
G_AB = Generator()
G_BA = Generator()
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB.cuda()
    G_BA.cuda()
    D_A.cuda()
    D_B.cuda()
    cycle_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

if opt.load_model != '':
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, load_model)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, load_model)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (opt.dataset_name, load_model)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (opt.dataset_name, load_model)))
    optimizer_G.load_state_dict(torch.load("saved_models/%s/optimizer_G_%s.pth" % (opt.dataset_name, opt.load_model)))
    optimizer_D_A.load_state_dict(torch.load("saved_models/%s/optimizer_D_A_%s.pth" % (opt.dataset_name, opt.load_model)))
    optimizer_D_B.load_state_dict(torch.load("saved_models/%s/optimizer_D_B_%s.pth" % (opt.dataset_name, opt.load_model)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Image transformations
random_transforms_ = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1),
]
transforms_ = [
    transforms.Resize(int(opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset_Pixellated(
        "../../data/%s/%%s/A" % opt.dataset_name,
        transforms_=transforms_,
        random_transforms_=random_transforms_,
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    ImageDataset_Pixellated(
        "../../data/%s/%%s/A" % opt.dataset_name,
        mode="test",
        transforms_=transforms_,
        random_transforms_=random_transforms_,
    ),
    batch_size=4,
    shuffle=True,
    num_workers=opt.n_cpu,
)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    validity = D(interpolates)
    fake = Variable(FloatTensor(np.ones(validity.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=validity,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_images(batches_done):
    """Saves a generated sample from the test set"""
    imgs = next(iter(val_dataloader))
    real_A = Variable(imgs["A"].type(FloatTensor))
    fake_B = G_AB(real_A)
    real_B = Variable(imgs["B"].type(FloatTensor))
    fake_A = G_BA(real_B)
    AB = torch.cat((real_A.data, fake_B.data), -2)
    BA = torch.cat((real_B.data, fake_A.data), -2)
    img_sample = torch.cat((AB, BA), -2)
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=8, normalize=True)

if cuda:
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


# ----------
#  Training
# ----------

batches_done = 0
prev_time = time.time()
for epoch in range(opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Configure input
        imgs_A = Variable(batch["A"].type(FloatTensor))
        imgs_B = Variable(batch["B"].type(FloatTensor))

        # ----------------------
        #  Train Discriminators
        # ----------------------

        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()

        # Generate a batch of images
        fake_A = G_BA(imgs_B).detach()
        fake_B = G_AB(imgs_A).detach()

        # ----------
        # Domain A
        # ----------

        # Compute gradient penalty for improved wasserstein training
        gp_A = compute_gradient_penalty(D_A, imgs_A.data, fake_A.data)
        # Adversarial loss
        D_A_loss = -torch.mean(D_A(imgs_A)) + torch.mean(D_A(fake_A)) + lambda_gp * gp_A

        # ----------
        # Domain B
        # ----------

        # Compute gradient penalty for improved wasserstein training
        gp_B = compute_gradient_penalty(D_B, imgs_B.data, fake_B.data)
        # Adversarial loss
        D_B_loss = -torch.mean(D_B(imgs_B)) + torch.mean(D_B(fake_B)) + lambda_gp * gp_B

        # Total loss
        D_loss = D_A_loss + D_B_loss

        D_loss.backward()
        optimizer_D_A.step()
        optimizer_D_B.step()

        if i % opt.n_critic == 0:

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Translate images to opposite domain
            fake_A = G_BA(imgs_B)
            fake_B = G_AB(imgs_A)

            # Reconstruct images
            recov_A = G_BA(fake_B)
            recov_B = G_AB(fake_A)

            # Adversarial loss
            G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
            # Cycle loss
            G_cycle = cycle_loss(recov_A, imgs_A) + cycle_loss(recov_B, imgs_B)
            # Total loss
            G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle

            G_loss.backward()
            optimizer_G.step()

            # --------------
            # Log Progress
            # --------------

            # Determine approximate time left
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / opt.n_critic)
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    D_loss.item(),
                    G_adv.data.item(),
                    G_cycle.item(),
                    time_left,
                )
            )

        # Check sample interval => save sample if there
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

        batches_done += 1

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (opt.dataset_name, epoch))
        torch.save(optimizer_G.state_dict(), "saved_models/%s/optimizer_G_%d.pth" % (opt.dataset_name, epoch))
        torch.save(optimizer_D_A.state_dict(), "saved_models/%s/optimizer_D_A_%d.pth" % (opt.dataset_name, epoch))
        torch.save(optimizer_D_B.state_dict(), "saved_models/%s/optimizer_D_B_%d.pth" % (opt.dataset_name, epoch))
