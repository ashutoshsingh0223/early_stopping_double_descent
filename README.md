# Early_stopping_double_descent
This repository contains the code for reproducing figures and results in the paper ``Early Stopping in Deep Networks: Double Descent and How to Eliminate it''.

# Requirements
The following Python libraries are required to run the code in this repository:

```
numpy
jupyter
torch
torchvision
```
and can be installed with `pip install -r requirements.txt`.

For this code to run it requires the CIFAR dataset tranformed to an `.npz` file and located in `./datasets` inside the root directory. CIFAR dataset does not come in that format and has a separate raw format. You need to download and convert the dataset before running for `MCNN`(5 layer conc neural net). Run the following to achieve this - 

`python3 get_dataset.py`

# Usage
All the figures in the paper can be reproduced by running the respective notebooks as indicated below:

**Figure 2**: Bias-variance trade-off curves for the linear regression model can be reproduced by running the `sum_bias_variance_tradeoffs` notebook.

**Figure 3**: Double descent in the two layer neural network and the elimination of the double descent through the scaling of the stepsizes of the two layers can be reproduced by running the `early_stopping_two-layer-nn_double_descent.ipynb` notebook.

**Figure 1-a, 4**: Double descent in the 5-layer convolutional network and the elimination of the double descent through the scaling of the stepsizes of the different layers can be reproduced by running the `early_stopping_deep_double_descent.ipynb` notebook. 

The numerical results can be reproduced by training the 5-layer convolutional network with `python3 train.py --config $CONFIG_FILE` where `CONFIG_FILE` points to the `config.json` file of the desired setup in the `./results/` directory.

* NOTE: * Please set the param `gpu` in the configs accordingly. Mostly it will be `0` but you might get errors with the default setting. Please change it accordingly.

## Disclaimers
**Figure 1-a, 7**: The bias and variance is measured as proposed in [Yang et al. \[2020\]](https://github.com/yaodongyu/Rethink-BiasVariance-Tradeoff) but adopted to measure bias-variance at each epoch. This may result in highly noisy measurements for the early training phase (see [this notebook](notebooks/early_stopping_deep_double_descent.ipynb) for details).

# Reproducability Challenge

The results of the the paper were reproduced for the ML Reproducability Challenge 2021. We reproduce the results for Linear Model, Two-Later NN, 5-Layer CNN and ResNet-18.
As a part of this experiment we make it easier to reproduce the results and also fix and update the original code at many places. Please follow the steps in Usage section by original authors to reproduce their results or you can use our contributions as well to comnpare the reproduced results with the original results.

**NOTE**: For all out experiments the configs we use have `secure_checkpoint=True` and `gpu=null` this means that we save checkpoint after x number of epochs and use all the GPUs available on the machine.
We had to modify/fix the configs of original authors at few to get their code working.

### Linear Model
We use the notebook by original authors `sum_bias_variance_tradeoffs` to reproduce for this model

### Two-Layer NN
See this notebook on [google collab](https://colab.research.google.com/drive/1hfB_j5WygqFeSwb_DygXsd-ddohUAAMu?usp=sharing). In this notebook we train a two-layer with different weight scales for both layers and for each weight-scale config we train the network twice; with and without learning scaling. We also plot the results of orginal authors and the reproduced results.

### MCNN
See this notebook on [google collab](https://colab.research.google.com/drive/13JDAAPpuScKt-_37A0PNob8llDr0AenW?usp=sharing).
Training the MCNN for more than 500 epochs might not be possible on google collab(atleast the free version). Here we first trained the MCCN for 2k epochs on a separate machine and then use this notebook to plot the results.

### ResNet-18


## Citation
```
@article{heckel_yilmaz_2020,
    author    = {Reinhard Heckel and Fatih Furkan Yilmaz},
    title     = {Early Stopping in Deep Networks: Double Descent and How to Eliminate it},
    journal   = {arXiv:2007.10099},
    year      = {2020}
}
```

## Licence

All files are provided under the terms of the Apache License, Version 2.0.