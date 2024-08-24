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

Extra utils for pickle manipulations and tensor reshaping; calculating the accuracy over FER and BER; several constants; and the config singleton class.
The config works by the [singleton design pattern](https://en.wikipedia.org/wiki/Singleton_pattern). Check the link if unfamiliar. 

The config is accessible from every module in the package, featuring the next parameters:
1. seed - random number generator seed. Integer.
2. channel_type - run either siso or mimo setup. Values in the set of ['SISO','MIMO']. String.
3. channel_model - chooses the channel taps values, either synthetic or based on COST2100. String in the set ['Cost2100','Synthetic'].
4. detector_type - selects the training + architecture to run. Short description of each option: 
* 'joint_black_box - Joint training of the black-box fully connected detector in the MIMO case.
* 'online_black_box' - Online training of the black-box fully connected detector in the MIMO case.
* 'joint_deepsic' - Joint training of the DeepSIC detector in the MIMO case.
* 'online_deepsic' - Online training of the DeepSIC detector in the MIMO case.
* 'meta_deepsic' - Online meta-training of the DeepSIC detector in the MIMO case.
* 'joint_rnn' - Joint training of the RNN detector in the SISO case.
* 'online_rnn' - online training of the RNN detector in the SISO case.
* 'joint_viterbinet' - Joint training of the ViterbiNet equalizer in the SISO case.
* 'online_viterbinet' - Online training of the ViterbiNet equalizer in the SISO case.
* 'meta_viterbinet' - Online meta-training of the ViterbiNet equalizer in the SISO case.
5. linear - whether to apply non-linear tanh at the channel output, not used in the paper but still may be applied. Bool.
6.fading_in_channel - whether to use fading. Relevant only to the synthetic channel. Boolean flag.
7. snr - signal-to-noise ratio, determines the variance properties of the noise, in dB. Float.
8. modulation_type - either 'BPSK' or 'QPSK', string.
9. memory_length - siso channel hyperparameter, integer.
10. n_user - mimo channel hyperparameter, number of transmitting devices. Integer.
11. n_ant - mimo channel hyperparameter, number of receiving devices. Integer.
12. block_length - number of coherence block bits, total size of pilot + data. Integer.
13. pilot_size - number of pilot bits. Integer.
14. blocks_num - number of blocks in the tranmission. Integer.
15. loss_type - 'CrossEntropy', could be altered to other types 'BCE' or 'MSE'.
16. optimizer_type - 'Adam', could be altered to other types 'RMSprop' or 'SGD'.
17. joint_block_length - joint training hyperparameter. Offline training block length. Integer.
18. joint_pilot_size - joint training hyperparameter. Offline training pilots block length. Integer.
19. joint_blocks_num - joint training hyperparameter. Number of blocks to train on offline. Integer.
20. joint_snrs - joint training hyperparameter. Number of SNRs to traing from offline. List of float values.
21. aug_type - what augmentations to use. leave empty list for no augmentations, or add whichever of the following you like: ['geometric_augmenter','translation_augmenter','rotation_augmenter']
22. online_repeats_n - if using augmentations, adds this factor times the number of pilots to the training batch. Leave at 0 if not using augmentations, if using augmentations try integer values in 2-5.

## resources

Keeps the COST channel coefficients vectors. Also holds config runs for the paper's numerical comparisons figures.

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
