import collections
import os
from abc import abstractmethod
from typing import Callable

import torch
from typing_extensions import Self


class Base(torch.nn.Module):

    @abstractmethod
    def save_model(self, path: str):
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path: str, device) -> Self:
        pass
