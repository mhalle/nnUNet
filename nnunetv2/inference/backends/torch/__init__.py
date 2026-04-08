"""PyTorch inference backend.

The two pieces of this backend are:

* :class:`ModelBundle` — filesystem + config: discovers a trained-model folder,
  parses ``plans.json`` / ``dataset.json``, builds the network, loads weights,
  owns preprocessing parameters. No inference compute.
* (forthcoming) ``InferenceEngine`` — pure compute over a ``ModelBundle``.

These mirror the MLX inference port's ``ModelBundle`` / ``InferenceEngine``
split so a single ``Protocol`` can describe both backends.

Note: this subpackage is named ``torch`` but should always be imported as
``nnunetv2.inference.backends.torch`` to avoid shadowing the real ``torch``
package. Code inside this subpackage aliases the real torch as ``import torch
as _torch`` if it needs both.
"""

from nnunetv2.inference.backends.torch.bundle import ModelBundle

__all__ = ["ModelBundle"]
