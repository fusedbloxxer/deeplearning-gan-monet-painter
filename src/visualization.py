from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from typing import Dict, List
from torch.utils import data
from IPython import display
from stats import GANStats
from model import GAN
from torch import nn
import numpy as np
import torch


def get_grid_samples(dataset, n_samples=25, shuffle=False, padding: int = 5, normalize: bool = True) -> torch.Tensor:
    """
        Retrieve a batch of samples from the dataset and return a grid.
    """
    loader = data.DataLoader(dataset, batch_size=n_samples, shuffle=shuffle)
    batch = next(iter(loader))
    nrows = int(torch.sqrt(torch.tensor(n_samples)).item())
    grid = make_grid(batch, nrow=nrows, padding=padding, normalize=normalize)
    grid = grid.permute((1, 2, 0))
    return grid


def get_grid_fakes(gan: GAN, noise: torch.Tensor = None,
              n_samples: torch.Size = torch.Size((25,)),
              normalize: bool = True,
              padding: int = 5) -> torch.Tensor:
    """
        Given a GAN and a noise tensor return a grid of fake samples.
        If no noise tensor is provided a random batch will be generated.
        The output of the GAN is expected to be in the range [-1, 1].
    """
    # Retain the previous training mode
    previous_mode: bool = gan.training

    # Use evaluation mode and no_grad to preview the output
    gan.eval()
    with torch.no_grad():
        # Pass noise through the generator
        if noise is None:
            fake_imgs = gan.G.generate(n_samples)
            noise = n_samples
            cells = n_samples[0]
        else:
            fake_imgs = gan.G(noise)
            cells = noise.size(0)

        # Assemble the fake samples into a grid
        fake_imgs = (0.5 * fake_imgs + 0.5)
        nrows = int(torch.sqrt(torch.tensor(cells)).item())
        grid = make_grid(fake_imgs, nrow=nrows, padding=padding, normalize=normalize)
        grid = grid.permute((1, 2, 0))

    # Reset the GAN back to its previous training mode
    gan.train(previous_mode)
    return grid


def display_weights(net: nn.Module, title: str):
    named_params = net.named_parameters()
    np_param_names = []
    np_params = []

    for name, param in named_params:
        np_params.append(param.clone().detach().view(-1).cpu().numpy())
        np_param_names.append(name)

    fig = plt.figure(figsize=(40, 2.5))
    fig.suptitle(title, fontsize=32, y=1.20)
    count = len(np_param_names)

    for i in range(count):
        plt.subplot(1, count, i + 1)
        plt.hist(np_params[i], bins=25)
        plt.title(np_param_names[i])
    plt.show()


def display_grad_history(grads: Dict[str, List[float]], title: str):
    fig = plt.figure(figsize=(40, 2.5))
    fig.suptitle(title, fontsize=32, y=1.20)
    count = len(grads)

    for i, (name, grads) in enumerate(grads.items()):
        plt.subplot(1, count, i + 1)
        plt.plot(np.arange(len(grads)), grads)
        plt.title(name)
        plt.xlabel('Epoch')
    plt.show()


def display_grad_hist(net: nn.Module, title: str):
    named_params = net.named_parameters()
    np_param_names = []
    np_params = []

    for name, param in named_params:
        np_params.append(param.grad.clone().detach().view(-1).cpu().numpy())
        np_param_names.append(name)

    fig = plt.figure(figsize=(40, 2.5))
    fig.suptitle(title, fontsize=32, y=1.20)
    count = len(np_param_names)

    for i in range(count):
        plt.subplot(1, count, i + 1)
        plt.hist(np_params[i], bins=25)
        plt.title(np_param_names[i])
    plt.show()


def display_stats(stats: GANStats, gan: GAN, epoch: int, fixed_noise: torch.Tensor) -> None:
    # Extract the stats
    loss_d, loss_g = stats.get_loss()
    prob_real, prob_fake1, prob_fake2 = stats.get_prob()

    # Print statistics
    display.clear_output(wait=False)
    print(f'Epoch {epoch}, D_prob_real={prob_real[-1]:.2}, D_prob_fake1={prob_fake1[-1]:.2},',
            f'D_prob_fake2={prob_fake2[-1]:.2}, D_loss={loss_d[-1]:.2}, G_loss={loss_g[-1]:.2}')

    # Show progression
    fig, axis = plt.subplots(1, 4, figsize=(40, 7))

    # Plot the losses
    axis[0].plot(np.arange(epoch + 1), loss_d, 'r', label='D_loss')
    axis[0].plot(np.arange(epoch + 1), loss_g, 'b', label='G_loss')
    axis[0].set_title('GAN Loss Evolution')
    axis[0].set_xlabel('Epoch')
    axis[0].set_ylabel('Loss')
    axis[0].legend()

    # Plot the probabilities
    axis[1].plot(np.arange(epoch + 1), prob_real, 'r', label='D_prob_real')
    axis[1].plot(np.arange(epoch + 1), prob_fake1,
                    'b', label='D_prob_fake1')
    axis[1].plot(np.arange(epoch + 1), prob_fake2,
                    'g', label='D_prob_fake2')
    axis[1].set_title('GAN Probability Evolution')
    axis[1].set_ylabel('Probability')
    axis[1].set_xlabel('Epoch')
    axis[1].legend()

    # Plot the fixed noise output
    grid_fakes = get_grid_fakes(gan, fixed_noise)
    axis[2].set_title('Fixed Noise Output')
    axis[2].imshow(grid_fakes.cpu())
    grid_randn = get_grid_fakes(gan)

    # Plot random noise output
    axis[3].set_title('Random Noise Output')
    axis[3].imshow(grid_randn.cpu())
    plt.show()

    # Plot the weight values for each layer
    display_weights(gan.D, title='Discriminator Weights')
    display_weights(gan.G, title='Generator Weights')

    # Display the Gradient flow evolution graph
    grads_d, grads_g = stats.get_grad()
    display_grad_history(grads_d, 'Discriminator Gradient Evolution')
    display_grad_history(grads_g, 'Generator Gradient Evolution')

    # Plot the current gradient histogram
    display_grad_hist(gan.D, title='Discriminator Gradient Histogram')
    display_grad_hist(gan.G, title='Generator Gradient Histogram')
