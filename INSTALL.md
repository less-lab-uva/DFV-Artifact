# Installation

To run the code, first make sure you have the following packages installed:
- python 3.7
- python 3 venv
- gcc 7+

Then begin the installation by running `./install.sh`. This will set up a python virtual environment and install the necessary dependencies. After install, run `source ./activate.sh` to activate the virtual environment.

`./install.sh` will install DNNV and DNNF. For DNNV, three verifiers will be installed: Neurify, nnenum and Verinet. After the installation ends, you can check if DNNV was installed correctly by running `dnnv --help`. You should see the name of the 3 verifiers under the **'verifiers'** section. Also you can check if DNNF was installed correctly by running `dnnf --help`.

`Verinet` uses `gurobi`, which requires a license, you can get an academic license for free at https://www.gurobi.com/academia/academic-program-and-licenses/. Once you have your license code, move to the DNNV folder that was created in the root of this repository, and execute

`bin/gurobi902/linux64/bin/grbgetkey *License Code*`

Lastly and before moving to the next step, you will need to unzip `rq1.zip`, `rq2.zip` and `rq3.zip`.

# Reproducing the study

## RQ 1

This directory contains the code to replicate the evaluation of our first research question. In this first experiment, we quantitatively and qualitatively assessed the effectiveness of DFV and its costs when applied in conjunction with 4 falsifiers (DeepFool, BIM, FGSM and PGD) and 3 verifiers (Neurify, nnenum and Verinet).

To recreate this experiment, you will need to do the following sequentially:

### 1) Training models
First, you will need to train the models. To do that, you can use `train_vae.py` to train the Fashion MNIST network and VAE. And use `train_vae_mrs.py` to train the VAE MRS. Models will be output to the directory `./saved_models/`. 

**WARNING!**

Training all the models can take a few hours, therefore we included all of them in the repository inside `./rq1/saved_models/`. Instead of running the above scripts, you can simply use the models that we included and move on to the next step.

### 2) Running Verifiers
Second, you will need to run the 3 verifiers. To do that, you can use `verify.sh` to verify all the properties on the Fashion MNIST network with and without DFV. By default the script will run all the tools used in the experiment. However a specific tool can be specified by passing `neurify`, `nnenum`, `verinet` as the first argument. Logs and counter-examples will be output to the directory `./output/Verifiers/`.

**WARNING!**

We have 20 properties, with a timeout of 1 hour each, further each property is run 5 times for the network with and without DFV. Therefore, in the case that all properties timeout, it will take up to 200 hours for each verifier. Since it takes a long time, we included the logs and counter-examples generated by DNNV in the repository inside `./rq1/output/Verifiers/`. Instead of running the above script, you can simply use the DNNV' outputs that we included and move on to the next step.

### 3) Running Falsifiers
Third, you will need to run the 4 falsifiers. To do that, you can use `falsify.sh` to falsify all the properties on the Fashion MNIST network with and without DFV. By default the script will run all the tools used in the experiment. However a specific tool can be specified by passing `deepfool`, `bim`, `fgsm`, `pgd` as the first argument. Logs and counter-examples will be output to the directory `./output/Falsifiers/`.

**WARNING!**

In this case, besides the timeout is set to 1 hour, DeepFool, BIM and FGSM will only do one attempt and will terminate if they do not find any counter-example. Therefore, the results for these falsifiers will finish relatively quickly. However PGD does multiple attempts and could potentially be running for one hour for each property, so it can take up to 200 hours to finish. Since running it could last a long time, we included the logs and counter-examples generated by DNNF in the repository inside `./rq1/output/Falsifiers/`. Instead of running the above script, you can simply use the DNNF' outputs that we included and move on to the next step.

### 4) Processing the data
The data generated by `verify.sh` in `./output/Verifiers` and `falsify.sh` in `./output/Falsifiers` need to be processed. In this step we parse the logs from DNNV and DNNF to extract useful information and calculate some metrics like MSE and SSIM, then everything is saved in pickle files (.pkl). You can do this by running `process_data.py`. The processed data will be output to the directory `./processed_data/`.

**WARNING!**

Processing the data can take a few hours, so we included the processed files in the repository inside `./rq1/processed_data/`. Instead of running the above script, you can simply use the pickle files that we included and move on to the next step.

### 5) Printing the study graphs
Once you have the processed files in the `./processed_data/` folder, you can generate the images used for the paper by running `print_images.py`. Images will be output to the directory `./images/`.

## RQ 2

This directory contains the code to replicate the evaluation of our second research question. In this second experiment, we explored the effects of the VAE’s latent space size, number and size of layers, and radius from the center of the latent space distribution on the efficacy of DFV.

To recreate this experiment, you will need to do the following sequentially:

### 1) Training models
First, you will need to train all the models. To do that, you can use `train_models.py` to train the Fashion MNIST network and the 90 Fashion MNIST VAEs. Use `train_vae_mrs.py` to train the VAE MRS. Models will be output to the directory `./saved_models/`. 

**WARNING!**

Training all the models will take several hours since each one of the 90 VAEs are trained for a 100 epochs. Since this process is going to take a long time, we included all the models in the repository inside `./rq2/saved_models/`. Instead of running the above script, you can simply use the models that we included and move on to the next step.

### 2) Running PGD Falsifier
Second, you will need to run PGD. To do that you can use `falsify_multidim.sh` to falsify all the properties on the 90 Fashion MNIST models with DFV. And then you can use `falsify_ls.sh` to falsify all the properties on the Fashion MNIST DFV 8-2-256 model, varying the radius of the latent space. Logs and counter-examples from `falsify_multidim.sh` will be output to the directory `./output/multidim_study/`, and the results from `falsify_ls.sh` will be output to the directory `./output/ls_study/`.

**WARNING!**

Although the timeout for each falsification problem is set to 5 minutes and we only use PGD; we now have 90 models with DFV and 20 properties that are run 5 times each. Consequently, in case that all properties timeout, it can take up to 750 hours. Since running it could last a long time, we included the logs and counter-examples generated by DNNF-PGD in the repository inside `./rq2/output/`. Instead of running the above scripts, you can simply use the DNNF' output that we included and move on to the next step.

### 3) Processing the data
The data generated by `falsify_multidim.sh` and `falsify_ls.sh` need to be processed. In this step we parse the logs from DNNF-PGD to extract useful information and calculate some metrics like MSE and SSIM, then everything is saved in pickle files (.pkl). You can do this by running `process_data.py`. The processed data will be output to the directory `./processed_data/`.

**WARNING!**

Processing the data can take a few hours, so we included the processed files in the repository inside `./rq2/processed_data/`. Instead of running the above script, you can simply use the pickle files that we included and move on to the next step.

### 4) Printing the study graphs
Once you have the processed files in the `./processed_data/` folder, you can generate the images used for the paper by running `print_images.py`. Images will be output to the directory `./images/`.

## RQ 3
This directory contains the code to replicate the evaluation of our third research question. In this third experiment, we assessed the scalability of DFV by applying it to a large DNN model for autonomous UAV control using 3 different input distribution models.

To recreate this experiment, you will need to do the following sequentially:

### 1) Training models
<!-- To train environment models, first download the pre-processed DroNet dataset by running `python download_data.py`. -->
Environment models can be trained by running `./train_fc.py` to train FC-*VAE_{DroNet}*, `./train_vae.py` to train Conv-*VAE_{DroNet}*, and `./train_dcgan.py` to train *GAN_{DroNet}*. Models will be output to the directory `./models/`.

**WARNING!**

Training all the models can take a few hours, therefore we included all of them in the repository inside `./rq3/models/`. Instead of running the above scripts, you can simply use the models that we included and move on to the next step.

### 2) Running the Falsifier
The PGD falsifier can then be run on the DroNet network, both without DFV and with DFV using the VAE and GAN models by running `./falsify.sh`. By default this script will run all 3 methods. A specific treatment can be specified by passing `dnn`, `vae`, or `gan` as the first argument to this script. If the first argument is `all` then all treatments will be run. This script will save all counter-examples to the directory `cex/`. The second argument to `./falsify.sh` can be used to specify a different name for this directory. The third argument accepted by this script specifies a timeout for each job in seconds. By default this timeout is 1 hour, or 3600 seconds. The fourth argument specifies where to save the logs, by default this is the directory `logs/falsification_logs`.

**WARNING!**

Running PGD on the different properties will take a few hours. Therefore, we included the logs and counter-examples generated by DNNF-PGD in the repository inside `./rq3/logs/falsification_logs`. Instead of running the above script, you can simply use the DNNF-PGD outputs that we included and move on to the next step.

### 3) Processing the Data
After running the falsifier, counter-examples can be converted to png formatted images and the plots for the paper can be generated by running `python npy_to_png.py cex logs/falsification_logs`, where `cex` is the directory containing the counter-examples and `logs/falsification_logs` is the directory containing the logs from running `falsify.sh`. This script will convert all counter-examples from the numpy `.npy` format to PNG images, will create CSV files containing the times and MRS values for each counter-example, and generate the plots shown in the paper.
