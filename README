# Structure

This code is organized as follows:
data: contains the file datamodule.py, which is the datamodule used to load the data and contains a few functions to preprocess the data.
configs: contains the configuration files for the different models, transforms and hyperparameters. 
    dataset: contains the configuration files for the different datasets. train_better is a manually curated dataset that contains the examples which we found to be most relevant for each class
    models: contains the configuration files for the different models. teachers and students are separated in different folders. 
    cross_validation is a boolean that indicates whether to use cross validation or not. If True, the dataset is split into 5 folds and the model is trained on each fold. If False, the model is trained on the whole dataset.

The core structure of the script is as follows:
run (and modify) experiment.py 's main function to carry out different experiments. In experiment.py we define a few functions, 
To separate training a teacher model with a few epochs as defined by warmup_epochs parameter, then we use the "bootstrapped" model to add pseudo-labels to the dataset and train a student model on the whole dataset.
We defined two functions to add pseudo labels, one which uses thresholding and one which selects the top k predictions for each class.

We also defined a function to train a model on the whole dataset without using pseudo-labels. This function is used to train the teacher model on the whole dataset with surrogate supervision signal of rotations applied. We then extract mid-level features from the teacher model and put a classifier on top of it. This classifier is trained on the training set and evaluated on the test set.

# Run:
Simply run "python experiment.py" in a shell. You can specify overrides as per the hydra syntax. Multiruns to test different parameters are also supported.
