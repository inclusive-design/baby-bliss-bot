# Notebooks

This directory contains all notebooks used for training or fine-tuning various models. Each notebook
usually comes with a accompanying `dockerfile.yml` to elaborate the environment that the notebook was running in.

## Texture Inversion

`texture_inversioni_flax.ipynb`: used to train stable diffusion model with texture inversion using Flax/JAX.
It refers to [this example of training texture inversion with Flax/JAX](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion#training-with-flaxjax). 
This training associates a few bliss symbol png files with a token `<bliss-symbol>`. After the fine-tuning, the
model didn't pick up any features of bliss symbols. This experiment is more for learning machine training and Azure.

**Note**: when using Azure to run this job, the sensitive information for connecting to the Azure subscription in the
section 1 needs to be filled in before running. If using other platforms, this section should be replaced with the
credential verification for that platform.

`texture_inversion_dockerfile.yml`: the docker environment used for running this training.

**Conclusion**: Texture inversion model is not useful in generating new Bliss symbols for brainstorming.
