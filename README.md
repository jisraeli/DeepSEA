Training DeepSEA deep convolutional network model and variant classifiers
============================================================================
DEPENDENCIES

1.	Install CUDA driver. A high-end NVIDIA CUDA compatible graphics card with enough memory is required to train the model. I use Tesla K20m with 5Gb memory for training the model.

2.	Installing torch and basic package dependencies following instructions from 
http://torch.ch/docs/getting-started.html
You may need to install cmake first if you do not have it installed. It is highly recommended to link against OpenBLAS or other optimized BLAS library when building torch, since it makes a huge difference in performance while running on CPU.

3.	Install torch packages required for training only: cutorch, cunn, mattorch. You may install through `luarock install [PACKAGE_NAME]` command. Note mattorch requires matlab. If you do not have matlab, you may try out https://github.com/soumith/matio-ffi.torch and change 1_data.lua to use matio instead (IMPORTANT: if you use matio, place remove the ":tr\
anspose(3,1)" and "transpose(2,1)" operation in 1_data.lua. The dimesions have been correctly handled by matio.).

============================================================================
Usage Example 

th main.lua -save results

The output folder will be under ./results . The folder will inlcude the model file as well as log files for monitoring training progress.

You can specify various parameters for main.lua e.g. set learning rate by -LearningRate. Take a look at main.lua for the options. 

Short explanation of the code: 1_data.lua reads the training and validation data; 2_model.lua specify the model; 3_loss.lua specify the loss function; 4_train.lua do the training.




============================================================================
Data

ENCODE and Roadmap Epigenomics data were used for labeling and the HG19 human genome was used for input sequences. The data is splitted to training, validation and test sets. The genomic regions are splitted to 200bp bins and labeled according to chromatin profiles. We kept the bins that have at least one TF binding event (note that TF binding event is measured by any overlap with a TF peak, not the >50% overlap criterion used for labeling).