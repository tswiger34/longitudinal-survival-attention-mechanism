from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, TypeVar, Union

import torch
import torchvision
from torchvision.models import ResNet, SwinTransformer

ModelArchitecture = Union[ResNet, SwinTransformer]

_TEncoderArchitecture = TypeVar("_TEncoderArchitecture", bound=ModelArchitecture)


class ImageEncoder(Protocol[_TEncoderArchitecture]):
    weights: str = "IMAGENET1K_V1"
    model_type: type[_TEncoderArchitecture]

    def __init__(self):
        self.set_model_state()

    @property
    @abstractmethod
    def model(self) -> _TEncoderArchitecture:
        """The initialized encoder model with IMAGENET1K_V1 weights"""
        pass

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Number of input features in the original encoder model"""
        pass

    @abstractmethod
    def set_model_state(self) -> None:
        """Sets the internal model state of the classifier/final layer by calling `torch.nn.Identity`"""


class ResNetEncoder(ImageEncoder[ResNet]):
    def __init__(self):
        super().__init__()

    @property
    def model(self):
        return torchvision.models.resnet18(weights=self.weights)

    @property
    def n_features(self):
        return self.model.fc.in_features

    def set_model_state(self):
        self.model.fc = torch.nn.Identity()


class SwinEncoder(ImageEncoder[SwinTransformer]):
    def __init__(self):
        super().__init__()

    @property
    def model(self):
        return torchvision.models.swin_v2_t(weights=self.weights)

    @property
    def n_features(self):
        return self.model.head.in_features

    def set_model_state(self):
        self.model.head = torch.nn.Identity()
