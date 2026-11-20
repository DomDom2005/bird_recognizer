"""Model registry: model name -> factory (SPEC §5, §15.9).

The CLI and training loop never import concrete model classes; they ask the registry.
Stage 5 registers transfer and CRNN models here.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .models.cnn import SmallCNN

ModelFactory = Callable[[int], nn.Module]

_FACTORIES: dict[str, ModelFactory] = {
    "cnn": lambda n_classes: SmallCNN(n_classes=n_classes),
}


def register_model(name: str, factory: ModelFactory) -> None:
    """Register (or replace) a model factory; used by later stages."""
    if not name:
        raise ValueError("nazwa modelu nie może być pusta")
    _FACTORIES[name] = factory


def available_models() -> list[str]:
    return sorted(_FACTORIES)


def build_model(name: str, n_classes: int) -> nn.Module:
    """Instantiate a model; raises ``ModelError`` for unknown names."""
    from .errors import ModelError

    factory = _FACTORIES.get(name)
    if factory is None:
        raise ModelError(f"nieznany model '{name}'; dostępne: {available_models()}")
    return factory(n_classes)
