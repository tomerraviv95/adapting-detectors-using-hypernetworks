# adapting-detectors-using-hypernetworks
*"A man, remember, whether rich or poor, should do something in this world. No one can find happiness without work."* 

--The Adventures of Pinocchio (by Carlo Collodi).

# Modular Hypernetworks for Scalable and Adaptive Deep MIMO Receivers

Python repository for the magazine paper "Modular Hypernetworks for Scalable and Adaptive Deep MIMO Receivers".

Please cite our [paper](https://arxiv.org/pdf/2408.11920), if the code is used for publishing research. 

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [datasets](#datasets)
    + [detectors](#detectors)
    + [plotting](#plotting)
    + [utils](#utils)
  * [resources](#resources)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This repository implements the modular hypernetwork for DeepSIC. We also implemented the joint and online learning described in the paper as baselines.
From the training prespective, you can choose between joint training (the receiver is trained offline in a pre-test phase, using data simulated from multitude of channel realizations and 
No additional training is done online in the test phase), online training (the receiver is trained online using the pilots batch at each time step) and the hypernetwork-based approach described in the paper.
You can simulate the methods over a range of different SNRs for two channels (synthetic and COST 2100 ones), using T blocks for each SNR point, or visualize the blockwise performance over a transmission of T blocks.

# Folders Structure

*Main Run* - Run the evaluate.py script  after choosing approriate hyperparameters and setup in the config.yaml to run the eval.

## python_code 

The python simulations of the simplified communication chain: symbols generation, data transmission and detection.

### datasets 

Under the channel folder, it includes the two channel models. Under the communication_blocks folder, we have all generation, modulation and transmission of data. This defines the pilots and information datasets as described in the paper. In users_network.py you can simulate users leaving and joining the network, such that the number of columns in the transmitted data matrix changed with time.

Finally, the data wrapper is in the channel_dataset.py file.

### detectors

The backbone detectors and their respective training are in this folder. Note that deepsic folder holds the backbone detector model, and the two trainer approaches, either online or joint. Under the hypernetwork directory we have the hyper deepsic implementation, which runs deepsic with the weights as input, the hypernetwork itself which is a simple sequence of feed-forward layers, and finally the hypernetwork trainer in hypernetwork_deepsic_trainer.py. Note that to train this module, I sampled batches containing blocks from all users configuration at each epoch. I have tried other sampling methods, and they all fail if one does not sample and train on all users configuration simultaneously. 

Each trainer is executable by running the 'evaluate.py' script after adjusting the config.yaml hyperparameters and choosing the desired method.

### plotting

Features main plotting tools for the paper:

* snrs_plotter - plotting of the SNRs per user for the synthetic channel, as in Fig. 2 in the paper.
* plot_ser_vs_block_index.py - plotting the SER as a function of the block index, as in Fig. 3-6 in the paper. 
* plot_ser_vs_cost_snr.py - plotting the SER versus the cost snr, as in Fig. 7.
* plot_ser_vs_pilots.py - plotting the SER versus the pilots number, as in Fig. 8.

### utils

Extra utils for pickle manipulations, metric calculations, channel estimation and constants holding.
The config in config_singleton.py is used throughout the repo, and follows the [singleton design pattern](https://en.wikipedia.org/wiki/Singleton_pattern). Check the link if unfamiliar. 

The config.yaml is accessible from every module in the package, featuring the next parameters which should be changed before running eval/plotting:
1. seed - random number generator seed. Integer.
2. n_ant - integer, the number of antennas in the base station.
3. channel_type - Values in the set of ['SED','COST']. String.
4. cost_snr - float value for setting the snr in the cost channel only.
5. detector_type - selects the training + architecture to run. Values in ['online_deepsic','joint_deepsic','hyper_deepsic'].
6. train_test_mismatch - bool. If True, then if evaluating on channel_type of 'SED' training will be done on 'COST' and vice versa.
7. train_block_length - size of training block for joint training of either hypernetwork or joint deepsic. Integer.
8. test_block_length - size of total block for testing, composed of pilots + information part. Integer.
9. test_pilots_length - number of pilots, integer. The information size is set to test_block_length - test_pilots_length.
10. test_blocks_num - number of validation blocks, denoted as T in the paper. Integer.

## resources

Keeps the COST channel coefficients vectors.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 3060 with CUDA 12. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f environment.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the deep_ensemble environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\environment\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!
