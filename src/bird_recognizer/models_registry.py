"""Model registry: model name -> factory (SPEC §5, §15.10).

The CLI and the training loop never import concrete model classes; they ask the
registry and pass model options (``arch``, ``pretrained``, ``pool``, ``rnn``) as
keyword arguments. Adding a family = registering one factory here.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import nn

from .models.cnn import SmallCNN
from .models.crnn import CRNN
from .models.transfer import build_transfer

ModelFactory = Callable[..., nn.Module]

_FACTORIES: dict[str, ModelFactory] = {
    "cnn": lambda n_classes, **options: SmallCNN(n_classes=n_classes, **options),
    "crnn": lambda n_classes, **options: CRNN(n_classes=n_classes, **options),
    "transfer": lambda n_classes, **options: build_transfer(n_classes, **options),
}


def register_model(name: str, factory: ModelFactory) -> None:
    """Register (or replace) a model factory; used by tests and later stages."""
    if not name:
        raise ValueError("nazwa modelu nie może być pusta")
    _FACTORIES[name] = factory


def available_models() -> list[str]:
    return sorted(_FACTORIES)


def build_model(name: str, n_classes: int, **options: object) -> nn.Module:
    """Instantiate a model; raises ``ModelError`` for unknown names."""
    from .errors import ModelError

    factory = _FACTORIES.get(name)
    if factory is None:
        raise ModelError(f"nieznany model '{name}'; dostępne: {available_models()}")
    return factory(n_classes, **options) if options else factory(n_classes)
