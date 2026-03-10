from abc import ABC, abstractmethod
from typing import ClassVar

from torch import Tensor, nn
from torchvision.models import ResNet, ResNet18_Weights, Swin_V2_T_Weights, SwinTransformer, resnet18, swin_v2_t


class ImageEncoder[T: nn.Module](ABC):
    """Abstract interface for torchvision image encoders with classifier head removed."""

    def __init__(self) -> None:
        self._model: T = self._build_model()
        self._n_features: int = self._extract_n_features(model=self._model)
        self._set_model_state()

    @property
    def model(self) -> T:
        return self._model

    @property
    def n_features(self) -> int:
        return self._n_features

    @abstractmethod
    def _build_model(self) -> T:
        """Builds the pretrained encoder architecture."""
        ...

    @abstractmethod
    def _extract_n_features(self, model: T) -> int:
        """Returns classifier input feature size before replacing the head."""
        ...

    @abstractmethod
    def _set_model_state(self) -> None:
        """Replaces the classifier head with ``nn.Identity`` in-place."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class ResNetEncoder(ImageEncoder[ResNet]):
    weights: ClassVar[ResNet18_Weights] = ResNet18_Weights.IMAGENET1K_V1

    def _build_model(self) -> ResNet:
        return resnet18(weights=self.weights)

    def _extract_n_features(self, model: ResNet) -> int:
        return model.fc.in_features

    def _set_model_state(self) -> None:
        self._model.fc = nn.Identity()


class SwinEncoder(ImageEncoder[SwinTransformer]):
    weights: ClassVar[Swin_V2_T_Weights] = Swin_V2_T_Weights.IMAGENET1K_V1

    def _build_model(self) -> SwinTransformer:
        return swin_v2_t(weights=self.weights)

    def _extract_n_features(self, model: SwinTransformer) -> int:
        return model.head.in_features

    def _set_model_state(self) -> None:
        self._model.head = nn.Identity()
