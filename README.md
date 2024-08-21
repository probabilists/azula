![Azula's banner](https://raw.githubusercontent.com/probabilists/azula/master/docs/images/banner.svg)

# Azula - Diffusion models in PyTorch

Azula is a Python package that implements diffusion models in [PyTorch](https://pytorch.org). Its goal is to unify the different formalisms and notations of the generative diffusion models literature into a single, convenient and hackable interface.

> In the [Avatar](https://wikipedia.org/wiki/Avatar:_The_Last_Airbender) cartoon, [Azula](https://wikipedia.org/wiki/Azula) is a powerful fire and lightning bender ⚡️

## Installation

The `azula` package is available on [PyPI](https://pypi.org/project/azula), which means it is installable via `pip`.

```
pip install azula
```

Alternatively, if you need the latest features, you can install it from the repository.

```
pip install git+https://github.com/probabilists/azula
```

## Getting started

In Azula's formalism, a diffusion model is the composition of three elements: a noise schedule, a denoiser and a sampler.

* A noise schedule is a mapping from a time $t \in [0, 1]$ to the signal scale $\alpha_t$ and the noise scale $\sigma_t$ in a perturbation kernel $p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X_t, \sigma_t^2 I)$ from a "clean" random variable $X \sim p(X)$ to a "noisy" random variable $X_t$.

* A denoiser is a neural network trained to predict $X$ given $X_t$.

* A sampler defines a series of transition kernels $q(X_s \mid X_t)$ from $t$ to $s < t$ based on a noise schedule and a denoiser. Simulating these transitions from $t = 1$ to $0$ samples approximately from $p(X)$.

This formalism is closely followed in the `azula` module.

```python
from azula.denoise import PreconditionedDenoiser
from azula.noise import VPSchedule
from azula.sample import DDPMSampler

# Choose the variance preserving (VP) noise schedule
schedule = VPSchedule()

# Initialize a denoiser
denoiser = PreconditionedDenoiser(
    backbone=CustomNN(in_features=5, out_features=5),
    schedule=schedule,
)

# Train to predict x given x_t
optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

for x in train_loader:
    t = torch.rand((batch_size,))

    loss = denoiser.loss(x, t).mean()
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

# Generate 64 points in 1000 steps
sampler = DDPMSampler(denoiser.eval(), steps=1000)

x_1 = torch.randn((64, 5))
x_0 = sampler(x_1)
```

Alternatively, Azula's plugin interface allows to load pre-trained models and use them with the same convenient interface.

```python
import sys

sys.path.append("path/to/guided-diffusion")

from azula.plugins import adm
from azula.sample import DDIMSampler

# Download weights from openai/guided-diffusion
denoiser = adm.load_model("imagenet_256x256")

# Generate a batch of 4 images
sampler = DDIMSampler(denoiser, steps=64).cuda()

latents = torch.randn((4, 3 * 256 * 256)).cuda()
labels = torch.randint(1000, size=(4,)).cuda()

images = sampler(latents, y=labels)
images = images.reshape(4, 3, 256, 256)
```

For more information, check out the documentation and tutorials at [azula.readthedocs.io](https://azula.readthedocs.io).

## Contributing

If you have a question, an issue or would like to contribute, please read our [contributing guidelines](https://github.com/probabilists/azula/blob/master/CONTRIBUTING.md).
