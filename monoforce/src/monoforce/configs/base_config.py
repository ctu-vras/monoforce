from abc import ABC
import torch


class BaseConfig(ABC):
    """
    Base configuration class. This class is used to store the configuration of the simulation.
    """

    def to(self, device: torch.device | str):
        """
        Moves all tensors to the specified device.

        Args:
            device (torch.device): device to move the tensors to.

        Returns:
            None
        """
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))

        return self
