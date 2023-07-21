# Train StyleGAN2-ADA
This article documents training a StyleGAN2-ADA network with a set of Bliss symbols image files using Compute Canada' servers.

The basis of the work is the article [How to Train StyleGAN2-ADA with Custom Dataset](https://towardsdatascience.com/how-to-train-stylegan2-ada-with-custom-dataset-dc268ff70544).  While the following closely adheres to the article's recipe, the main difference is the use of PyTorch libraries instead of Tensorflow.  The instructions in the article are based on the Tensorflow implementation of StyleGAN2-ADA, but the [README](https://github.com/NVlabs/stylegan2-ada#readme) for the actual StyleGAN2-ADA project states that "The [Official PyTorch version](https://github.com/NVlabs/stylegan2-ada-pytorch) is now available and supersedes the TensorFlow version".  Where necessary, adjustments are made to fit with the PyTorch implementation.

## Set up on Compute Canada Clusters
First, set up the python virtual environment and install the requisite python packages for the PyTorch implementation of StyleGAN2-ADA.  The environment is created once inside the user's home directory so that multiple training runs can be executed at different times without having to re-install all the packages for each run.  That is, the results of this step are not output in the `scratch` directory, where the batch jobs are run.  The following directory structure was used for this work:  `~/BlissStyleGAN/StyleGAN2/`

Note that the `pip install` command uses the `--no-index` flag so as to not search PyPI repositories, but to use Compute Canada repositories where feasible, as recommended in their [wiki](https://docs.alliancecan.ca/wiki/Python#Installing_packages).

```
[StyleGAN2]$ module load python/3.7
[StyleGAN2]$ virtualenv --no-download pytorch
[StyleGAN2]$ source pytorch/bin/activate
[StyleGAN2]$ pip install --no-index --upgrade pip
[StyleGAN2]$ pip install --no-index torch==1.7.1
[StyleGAN2]$ pip install --no-index click
[StyleGAN2]$ pip install --no-index pillow
[StyleGAN2]$ pip install --no-index tqdm
[StyleGAN2]$ pip install --no-index requests
[StyleGAN2]$ pip install pyspng
[StyleGAN2]$ pip install --no-index ninja
[StyleGAN2]$ pip install imageio-ffmpeg==0.4.3
[StyleGAN2]$ pip install --no-index psutil
[StyleGAN2]$ pip install --no-index scipy
```

A `requirements.txt` file is provided, but for reference only.  It was not used to install the packages but was created after the above commands were all executed to keep a record of the versions of the packages in case StyleGAN2-ADA requires different versions or different packages in the future &mdash; `pip freeze --local > requirements.txt`

Secondly, clone the StyleGAN2-ADA pytorch source code, also placing it in the `~/BlissStyleGAN/StyleGAN2` directory:

```
[StyleGAN2]$ git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

Note that there is a bug in the image generation code when using grey-scale images for training &mdash; the kind of images that this project uses.  If you want to generate an image from a trained model, you will need to change the `generate.py` script.  [Issue 55](https://github.com/NVlabs/stylegan2-ada-pytorch/issues/55) has a proposed fix:

  - change line 121.  Line 121 is currently (as of 16-May-2023):

    ```
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')
    ```
  - change it to:

    ```
    img = np.squeeze(img, axis=3)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'L').save(f'{outdir}/seed{seed:04d}.png')
    ```

## Prepare the data
The set of images used to train the model must be in the correct form for use with StyleGAN2-ADA.  Its `dataset_tool.py` script is used to take a set of images, format, and output them to a directory.  The source images were taken from those created according to the [Prepare Image Set For StyleGAN models](./PrepareImageSetForStyleGAN.md) document.  These were gathered into a tar file and processed by an SBatch shell script to create a tar file of prepared images.  In summary:

  - Use the `def-styleGAN2AdaPytorchDataSetupBatch.sh` SBatch script to prepare the dataset.
  - The input tar file is named `blissSingleCharsGrey.tar`, in the user's home directory.
  - The output file is `preppedBliss4Pytorch.tar`, created in the user's home directory by the SBatch script.

The SBatch script itself must be copied to the user's `scratch` directory and is run from there using the `sbatch` command:

```
[scratch $] sbatch def-styleGAN2AdaPytorchDataSetupBatch.sh
```

SBatch jobs create a`SLURM_TMPDIR` temporary directory that holds all of the intermediate results of the job.  The final results are tarred by the batch script and copied back to the `StyleGAN2` folder in the home directory.  This is necessary since the `SLURM_TMPDIR` exists only for the duration of the SBatch job.  Once the job exits, the `SLURM_TMPDIR` is erased and ceases to exist.

Note that the `def-styleGAN2AdaPytorchDataSetupBatch.sh ` SBatch script is set to use the `def-whkchun` account and cluster.  Near the top of the script is a `module load ...` command for loading `cuda`.  The script can be used on the `ctb-whkchun` cluster, except that the Cuda `module load` command is different:

  - For `def-whkchun`, use:

    ```
    module load nixpkgs/16.09  intel/2018.3  cuda/10.0.130 cudnn/7.5
    ```
  - For `ctb-whkchun`, use:
    
    ```
    module load cuda cudnn
    ``` 

### Monitoring an SBatch Job
A log file is declared near the top of an SBatch script that is used by the system to output log messages as the job progresses.  The SBatch scripts used for data prepartion, training, and generation sets the log file name based on a job name, job ID, and the node used:

```
#SBATCH --output=%x-%N-%j.out  # %N for node name, %j for jobID
```

An example of a log file name is `StyleGAN-2-cdr112-1765758.out`.  This file can be opened from time to time to see how the job is progressing, or, if it failed, to use the error message to fix some aspect of the script.

## Run the training script

Run the `def-styleGAN2AdaPytorchTrainBatch.sh` SBatch script to train the system using the prepared data. As with the data preparation script, the training script is run from within the user's `scratch` directory.  And, while the prepared dataset is in the user's home directory, the output training results are in the `./scratch/pytorch-ada-results` directory.

The sub-directories within `pytorch-ada-results` are named according to the training run. Each sub-directory begins with a five digit count; the first training run is denoted by `00000`.  The remainder of the name is the name of the dataset plus some of the command line flags.  For example, the first run's subdirectory is `00000-preppedBlissSingleCharGrey-auto1`.  The second run is the `00001-preppedBlissSingleCharGrey-auto1-resumecustom` sub-directory.

At this point of development, there are some arguments within the `def-styleGAN2AdaPytorchTrainBatch.sh` script that must be modified by hand if the intent is to resume training based on the results of a previous training run:

  - If this is the first training run, the `train.py` command has no `--resume` argument.
  - If this is a second, third, etc. run, the `train.py` command has a `--resume` argument that points to the pickle file that contains the latest model from a previous run, e.g.:

    ```
    --resume="$OUTPUT_DIR/00001-preppedBlissSingleCharGrey-auto1-resumecustom/network-snapshot-000440.pkl"
    ```

To determine the exact name of the latest model file, explore the contents of the latest run subdirectory and find the <em>most recent</em> file with a name like `network-snapshot-000440.pkl`, where the digits differ from training run to training run.

WARNING:  Since the `scratch` directory and its contents are not permanent, once training is finished, move the results directory to the home directory at the earliest convenience.  One way to copy the results from within `scratch` is:

```
[scratch]$ cp -a -r pytorch-ada-results ~/BlissStyleGAN/StyleGAN2/pytorch-ada-results
```

## Generate New Images

Training runs end due to exceeding the job's time limit.  The training script will output a most recent ML model as a pickle file sometime before the job exits.  In fact, the training script outputs intermediate model files periodically during a single training run.

One can generate new images based on these models.  StyleGAN2-ADA's `generate.py` script is used to generate one or more images.  The `ctb-styleGAN2AdaPytorchGenerateBatch.sh` SBatch script can be used to generate images from a machine model pickle file.  It is run in the `scratch` directory and places the generated images in the home directory in `~/BlissStyleGAN/StyleGAN2/pytorch-ada-generate`:

```
[scratch]$ sbatch ctb-styleGAN2AdaPytorchGenerateBatch.sh
```

At this point in development, the SBatch script must be edited by hand to set the `--seed` arguments and have it point to the relevant model.  The following is an example.  The `--seed` argument is a sequence of comma separated numbers, or a range.  The model is the relevant pickle file, usually the latest file:

```
python ~/BlissStyleGAN/StyleGAN2/stylegan2-ada-pytorch/generate.py \
  --outdir="$OUTPUT_DIR" --trunc=0.5 --seeds=200,330,400 \
  --network="./pytorch-ada-results/00002-preppedBlissSingleCharGrey-auto1-resumecustom/network-snapshot-001640.pkl"
