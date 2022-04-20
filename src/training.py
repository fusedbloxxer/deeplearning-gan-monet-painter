from visualization import display_stats, get_grid_fakes
import matplotlib.pyplot as plt
from torch.utils import data
from IPython import display
from stats import GANStats
from typing import Tuple
from torch import optim
from model import GAN
from torch import nn
import numpy as np
import torch


def pretrain(gan: GAN, data_loader: data.DataLoader, optim_D: optim.Optimizer,
             device: torch.device, loss_fn: nn.Module, steps: int):
    # Pre-training the Discriminator
    it = iter(data_loader)
    N, _, _, _ = next(it).size()

    # Create the labels
    real_labels = torch.full((N,), 1.0, device=device)

    # Teach the Discriminator how the real distribution looks
    for step in range(steps):
        # Send the batch to the GPU
        real_imgs = next(it).to(device)

        # Classify the images
        optim_D.zero_grad()
        out_real = gan.classify(real_imgs)
        loss_real = loss_fn(out_real, real_labels)
        print(f'Pretrain Step {step} - Loss Real: {loss_real.detach().item()}',
              f'Prob Real: {out_real.detach().mean().item()}')

        # Propagate the loss and update the weights
        loss_real.backward()
        optim_D.step()


def train(num_epochs: int, epoch_size: int, gan: GAN, data_loader: data.DataLoader,
          optims: optim.Optimizer, d_steps: int, device: torch.device,
          loss_fn: nn.Module, fixed_noise: torch.Tensor,
          fake_label_noise: float = 0.0, real_label_noise: float = 0.0,
          pretraining: int = None):
    # Extract the optimizers
    optim_D, optim_G = optims

    # Store the stats
    stats = GANStats()

    # Get iterator over the data
    it = iter(data_loader)
    N, _, _, _ = next(it).size()

    # Use binary labels
    fake_labels = torch.full((N,), 0.0, device=device)
    real_labels = torch.full((N,), 1.0, device=device)

    # Add label noise
    fake_labels = fake_labels + fake_label_noise
    real_labels = real_labels - real_label_noise

    # Pretrain the Discriminator on the real images
    if pretraining:
        pretrain(gan, data_loader, optim_D, device, loss_fn, pretraining)

    # Train the DCGAN
    for epoch in range(num_epochs):
        for epoch_step in range(epoch_size):
            for d_step in range(d_steps):
                # Reset gradients
                optim_D.zero_grad()

                # Compute the output of the Discriminator for real images
                real_img = next(it).to(device)
                out_real = gan.D(real_img)
                loss_d_real = loss_fn(out_real, real_labels)
                loss_d_real.backward()

                # Compute the output of the Discriminator for fake images
                fake_img = gan.G.generate(torch.Size((N,)))
                out_fake = gan.D(fake_img.detach())
                loss_d_fake = loss_fn(out_fake, fake_labels)
                loss_d_fake.backward()

                # Update the model and retain the error
                stats.add_grad(net_d=gan.D)
                optim_D.step()
                loss_d = loss_d_real + loss_d_fake

                # Save the stats
                if d_step == d_steps - 1:
                    stats.add_loss(loss_d.detach())
                    stats.add_prob(out_real.detach(), out_fake.detach())

            # Reset gradients
            optim_G.zero_grad()

            # Compute the output of the Discriminator for fake images
            fake_img = gan.G.generate(torch.Size((N,)))
            out_fake = gan.D(fake_img)
            loss_g = loss_fn(out_fake, real_labels)

            # Update the model
            loss_g.backward()
            stats.add_grad(net_g=gan.G)
            optim_G.step()

            # Save the stats
            stats.add_loss(loss_g=loss_g.detach())
            stats.add_prob(prob_fake2=out_fake.detach())

        # Save the current stats
        stats.step()

        # Display the current stats
        if epoch % 3 == 0:
            display_stats(stats, gan, epoch, fixed_noise)
            print('LOSS D: ', loss_d.detach().item())
            print('LOSS G: ', loss_g.detach().item())
            print('P Real: ', out_real.mean().detach().item())
            print('P fake2: ', out_fake.mean().detach().item())
