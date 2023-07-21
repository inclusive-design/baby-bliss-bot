# Jobs

This directory contains all jobs used for training or fine-tuning various models.

## StyleGAN2-ADA
The [stylegan2-ada](./stylegan2-ada) folder contains:

   - `def-styleGan2AdaPytorchDataSetupBatch.sh` is the SBatch script for preparing the training dataset for StyleGAN2-ADA.  The script uses the `def-whkchun` cluster.
   - `def-styleGAN2AdaPytorchTrainBatch.sh` is the SBatch script for training. The script uses the `def-whkchun` cluster.
   - `ctb-styleGAN2AdaPytorchGenerateBatch.sh` is the SBatch script for generating an image from the StyleGAN2-ADA model.  The script uses the `ctb-whkchun` cluster.
   - `def-styleGAN2AdaPytorchGenerateBatch.sh` is the SBatch script that also can be used to generate images from the StyleGAN2-ADA model.  This version uses the `def-whkchun` cluster.
   - `requirements.txt` shows the packages used by the PyTorch implementation of StyleGAN2-ADA.  Note that this is not used to create the environment, but to document the environment after it was created.

See the [StyleGAN2-ADATraining.md](../docs/StyleGAN2-ADATraining.md) in the [documentation](../docs) folder for details on how to set up the environment.

## StyleGAN3

The [stylegan3](./stylegan3) directory contains:

   - `requirements.txt` is used with other module installations to set up the environment for training
   [the stylegan3 model](https://github.com/NVlabs/stylegan3) with the Bliss single characters.
   - `job_stylegan3.sh` is the job script submitted in [the Cedar platform](https://docs.alliancecan.ca/wiki/Cedar)
   to perform the training.

See the [TrainStyleGAN3Model.md](../docs/TrainStyleGAN3Model.md) in the [documentation](../docs) folder for details on
how to about how to train this model, training results and the conclusion about how useful it is.
