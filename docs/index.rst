.. image:: images/banner.svg
   :class: only-light

.. image:: images/banner_dark.svg
   :class: only-dark

Azula
=====

Azula is a Python package that implements diffusion models in `PyTorch <https://pytorch.org>`_. Its goal is to unify the different formalisms and notations of the generative diffusion models literature into a single, convenient and hackable interface.

Installation
------------

The :mod:`azula` package is available on `PyPI <https://pypi.org/project/azula>`_, which means it is installable via `pip`.

.. code-block:: console

   pip install azula

Alternatively, if you need the latest features, you can install it from the repository.

.. code-block:: console

   pip install git+https://github.com/probabilists/azula

Getting started
---------------

In Azula's formalism, a diffusion model is the composition of three elements: a noise schedule, a denoiser and a sampler.

* A noise schedule is a mapping from a time :math:`t \in [0, 1]` to the signal scale :math:`\alpha_t` and the noise scale :math:`\sigma_t` in a perturbation kernel :math:`p(X_t \mid X) = \mathcal{N}(X_t \mid \alpha_t X, \sigma_t^2 I)` from a "clean" random variable :math:`X \sim p(X)` to a "noisy" random variable :math:`X_t`.

* A denoiser is a neural network trained to predict :math:`X` given :math:`X_t`.

* A sampler defines a series of transition kernels :math:`q(X_s \mid X_t)` from :math:`t` to :math:`s < t` based on a noise schedule and a denoiser. Simulating these transitions from :math:`t = 1` to :math:`0` samples approximately from :math:`p(X)`.

This formalism is closely followed by Azula's API.

.. code-block:: python

   from azula.denoise import KarrasDenoiser
   from azula.noise import VPSchedule
   from azula.sample import DDPMSampler

   # Choose the variance preserving (VP) noise schedule
   schedule = VPSchedule()

   # Initialize a denoiser
   denoiser = KarrasDenoiser(
      backbone=CustomNN(in_features=5, out_features=5),
      schedule=schedule,
   )

   # Train to predict X given X_t
   optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-3)

   for x in train_loader:
      t = torch.rand((batch_size,))

      loss = denoiser.loss(x, t)
      loss.backward()

      optimizer.step()
      optimizer.zero_grad()

   # Generate 64 points in 1000 steps
   sampler = DDPMSampler(denoiser.eval(), steps=1000)

   x1 = sampler.init((64, 5))
   x0 = sampler(x1)

Alternatively, Azula's plugin interface allows to load pre-trained models and use them with the same convenient interface.

.. code-block:: python

   import sys

   sys.path.append("path/to/guided-diffusion")

   from azula.plugins import adm
   from azula.sample import DDIMSampler

   # Download weights from openai/guided-diffusion
   denoiser = adm.load_model("imagenet_256x256")
   denoiser.to("cuda")

   # Generate a batch of 4 images
   sampler = DDIMSampler(denoiser, steps=64)

   x1 = sampler.init((4, 3, 256, 256), device="cuda")
   x0 = sampler(x1)

   images = torch.clip((x0 + 1) / 2, min=0, max=1)

For more information, check out the :doc:`tutorials <../tutorials>` or the :doc:`API <../api>`.

.. toctree::
    :caption: azula
    :hidden:
    :maxdepth: 2

    tutorials.rst
    api.rst

.. toctree::
    :caption: Development
    :hidden:
    :maxdepth: 1

    Contributing <https://github.com/probabilists/azula/blob/master/CONTRIBUTING.md>
    Changelog <https://github.com/probabilists/azula/releases>
    License <https://github.com/probabilists/azula/blob/master/LICENSE>
