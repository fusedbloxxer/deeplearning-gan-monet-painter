from abc import abstractmethod, ABC
from typing import List, Tuple
import torch
import copy


class GANBatchStats():
    def __init__(self):
        # Confusion matrix
        self.__conf_matrix = torch.zeros((2, 2))

        # Losses
        self.__loss_d = []
        self.__loss_g = []

        # Discriminator predictions
        self.__prob_real = []
        self.__prob_fake1 = []
        self.__prob_fake2 = []

    def add_loss(self, loss_d: torch.Tensor = None, loss_g: torch.Tensor = None) -> None:
        if loss_d is not None:
            self.__loss_d.append(loss_d.item())

        if loss_g is not None:
            self.__loss_g.append(loss_g.item())

    def add_prob(self, prob_real: torch.Tensor = None,
                       prob_fake1: torch.Tensor = None,
                       prob_fake2: torch.Tensor = None) -> None:
        if prob_real is not None:
            self.__prob_real.append(prob_real.mean().item())

        if prob_fake1 is not None:
            self.__prob_fake1.append(prob_fake1.mean().item())

        if prob_fake2 is not None:
            self.__prob_fake2.append(prob_fake2.mean().item())

    def get_loss(self) -> Tuple[float, float]:
        return torch.tensor(self.__loss_d).mean().item(), \
            torch.tensor(self.__loss_g).mean().item()

    def get_prob(self) -> Tuple[float, float, float]:
        return torch.tensor(self.__prob_real).mean().item(),  \
            torch.tensor(self.__prob_fake1).mean().item(), \
            torch.tensor(self.__prob_fake2).mean().item()

    def get_conf_matrix(self) -> torch.Tensor:
        return self.__conf_matrix

    def reset(self):
        # Clear the state
        self.__loss_d.clear()
        self.__loss_g.clear()
        self.__prob_real.clear()
        self.__prob_fake1.clear()
        self.__prob_fake2.clear()
        self.__conf_matrix.fill_(0.0)


class GANStats():
    """
        Holds statistics for the GAN obtained during the training process.
        The step() method should be called after each training epoch, and the
        add_statistic method should be called after each batch.
    """

    def __init__(self):
        # Saved metrics
        self.__epoch_metrics: List[GANBatchStats] = []
        self.__batch_metrics = GANBatchStats()

    def add_loss(self, loss_d: torch.Tensor = None, loss_g: torch.Tensor = None) -> None:
        self.__batch_metrics.add_loss(loss_d, loss_g)

    def add_prob(self, prob_real: torch.Tensor = None,
                       prob_fake1: torch.Tensor = None,
                       prob_fake2: torch.Tensor = None) -> None:
        self.__batch_metrics.add_prob(prob_real, prob_fake1, prob_fake2)

    def get_loss(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.__epoch_metrics) == 0:
            return torch.tensor([]), torch.tensor([])

        loss_d = torch.zeros((len(self.__epoch_metrics),))
        loss_g = torch.zeros((len(self.__epoch_metrics),))

        for i, batch in enumerate(self.__epoch_metrics):
            loss_d[i], loss_g[i] = batch.get_loss()

        return loss_d, loss_g

    def get_prob(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(self.__epoch_metrics) == 0:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])

        prob_real = torch.zeros((len(self.__epoch_metrics),))
        prob_fake1 = torch.zeros((len(self.__epoch_metrics),))
        prob_fake2 = torch.zeros((len(self.__epoch_metrics),))

        for i, batch in enumerate(self.__epoch_metrics):
            prob_real[i], prob_fake1[i], prob_fake2[i] = batch.get_prob()

        return prob_real, prob_fake1, prob_fake2

    def get_conf_matrix(self) -> torch.Tensor:
        if len(self.__epoch_metrics) == 0:
            return torch.zeros((2, 2))

        return self.__epoch_metrics[-1].get_conf_matrix()

    def step(self):
        self.__epoch_metrics.append(copy.deepcopy(self.__batch_metrics))
        self.__batch_metrics.reset()

    def reset(self):
        self.__epoch_metrics.clear()
        self.__batch_metrics.reset()
