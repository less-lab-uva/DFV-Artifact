# DFV Artifact

This is the artifact to accompany the ASE technical track submission "Distribution Models for Falsification and Verification of DNNs".

Due to the size of trained models, we could not include them in this repository. The trained models used to evaluate DFV are available here: https://drive.google.com/drive/folders/1EPA-QaDw8SlpjAHLO4uT7WWdABIBLPtu

The artifact is split into several directories.

- Directory `rq1` contains the code and data from our experiments for research question 1. The environment models used to evaluate DFV, verifiers and falsifiers outputs, and processed data can be downloaded by running `download.py`.

- Directory `rq2` contains the code and data from our experiments for research question 2. The environment models used to evaluate DFV, falsifiers outputs, and processed data can be downloaded by running `download.py`.

- Directory `rq3` contains the code and data from our experiments for research question 3. The environment models used to evaluate DFV are available in the `RQ3 Models` directory at https://drive.google.com/drive/folders/1EPA-QaDw8SlpjAHLO4uT7WWdABIBLPtu.

To run the code, first run `./install.sh`. This will set up a python virtual environment and install the necessary dependencies. After install, run `source .venv/bin/activate` to activate the virtual environment.


## RQ1

### Models
Use `train_vae.py` to train the Fashion MNIST network and VAE. Use `train_vae_mrs.py` to train the VAE MRS.

### Verifiers
Use `verify.sh` to verify all the properties on the Fashion MNIST network with and withouth DFV. By default the script will run all the tools used in the experiment. However a specific tool can be specified by passing `neurify`, `nnenum`, `verinet` as the first arguemnt.

### Falsifiers
Use `falsify.sh` to falsify all the properties on the Fashion MNIST network with and withouth DFV. By default the script will run all the tools used in the experiment. However a specific tool can be specified by passing `deepfool`, `bim`, `fgsm`, `pgd` as the first arguemnt.

### Process data
The data generated by `verify.sh` and `falsify.sh` need to be processed. To do so, run `process_data.py`

### Print study graphs
The images used for the paper can be generated by running `print_images.py`

### Downloads
Running all the above scripts can be time-consuming, therefore by running `download.py` three folders will be downloaded `processed_data`, `saved_models` and `output`. These folders contain all the data needed to execute `print_images.py` without running the other scripts.


## RQ2

### Models
Use `train_models.py` to train the 90 Fashion MNIST VAEs. Use `train_vae_mrs.py` to train the VAE MRS.

### Falsifier - PGD
Use `falsify_multidim.sh` to falsify all the properties on the 90 Fashion MNIST models with DFV. Use `falsify_ls.sh` to falsify all the properties on the Fashion MNIST 8-2-256 model with DFV variating the radius of the latent space.

### Process data
The data generated by `falsify_multidim.sh` and `falsify_ls.sh` need to be processed. To do so, run `process_data.py`

### Print study graphs
The images used for the paper can be generated by running `print_images.py`

### Downloads
Running all the above scripts can be time-consuming, therefore by running `download.py` three folders will be downloaded `processed_data`, `saved_models` and `output`. These folders contain all the data needed to execute `print_images.py` without running the other scripts.


## RQ3

Environment models can be trained by running `./train_fc.py` to train FC-*VAE_{DroNet}*, `./train_vae.py` to train Conv-*VAE_{DroNet}*, and `./train_dcgan.py` to train *GAN_{DroNet}*.

The PGD falsifier can then be run on the DroNet network, both without DFV and with DFV using the VAE and GAN models by running `./falsify.sh`. By default this script will run all 3 methods. A specific treatment can be specified by passing `dnn`, `vae`, or `gan` as the first argument to this script. If the first argument is `all` then all treatments will be run. This script will save all counter-examples to the directory `cex/`. The second argument to `./falsify.sh` can be used to specify a different name for this directory. The third and final argument accepted by this script specifies a timeout for each job in seconds. By default this timeout is 1 hour, or 3600 seconds.

After running the falsifier, counter-examples can be converted to png formatted images and the plots for the paper can be generated by running `python npy_to_png.py cex logs`.
