from visualization import display_stats, get_grid_fakes
import matplotlib.pyplot as plt
from torch.utils import data
from IPython import display
from stats import GANStats
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
          loss_fn: nn.Module, fixed_noise: torch.Tensor, pretraining: int = None):
    # Extract the optimizers
    optim_D, optim_G = optims

    # Store the stats
    stats = GANStats()

    # Get iterator over the data
    it = iter(data_loader)
    N, _, _, _ = next(it).size()

    # Specify label types
    fake_labels = torch.full((N,), 0.0, device=device)
    real_labels = torch.full((N,), 1.0, device=device)

    # Pretrain the Discriminator on the real images
    if pretraining:
        pretrain(gan, data_loader, optim_D, device, loss_fn, pretraining)

    # Train the DCGAN
    for epoch in range(num_epochs):
        for _ in range(epoch_size):
            for _ in range(d_steps):
                # Reset gradients
                optim_D.zero_grad()

                # ^^^ Classify real images ^^^
                # Send data to GPU
                real_img = next(it).to(device)

                # Compute the output of the Discriminator for real images
                out_real = gan.D(real_img)
                loss_d_real = loss_fn(out_real, real_labels)

                # Accumulate gradients
                loss_d_real.backward()

                # ^^^ Classify fake images ^^^
                fake_img = gan.G.generate(torch.Size((N,)))

                # Compute the output of the Discriminator for fake images
                out_fake = gan.D(fake_img.detach())
                loss_d_fake = loss_fn(out_fake, fake_labels)

                # Update the model
                loss_d_fake.backward()
                optim_D.step()

                # Retain the error
                loss_d = loss_d_real + loss_d_fake

                # Save the stats
                stats.add_loss(loss_d)
                stats.add_prob(out_real.detach(), out_fake.detach())

            # --- Train the Generator ---
            # Reset gradients
            optim_G.zero_grad()

            # ^^^ Classify fake images ^^^
            fake_img = gan.G.generate(torch.Size((N,)))

            # Compute the output of the Discriminator for fake images
            out_fake = gan.D(fake_img)
            loss_g_fake = loss_fn(out_fake, real_labels)

            # Update the model
            loss_g_fake.backward()
            optim_G.step()

            # Save the stats
            stats.add_loss(loss_g=loss_g_fake)
            stats.add_prob(prob_fake2=out_fake.detach())

        # Save the current stats
        stats.step()

        # Display the current stats
        display_stats(stats, gan, epoch, fixed_noise)
