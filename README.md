# Miles per Gallon Prediction

Given a dataset containing the features (weigth, country, year, etc.) of several cars, compare the architectures of different Multi-Layer Perceptron and Radial Basis Function networks built with MATLAB and choose the one which gives the smallest error.

## Features Selection

Not all the features in the dataset are useful for the training of the MLPs and for this reason a subset of them have been choosen by using a Genetic Algorithm able compare the correlation of different subset of features with the MPG.

## Evaluation of the Networks

After choosing the features, a set of possible architectures for RBF and MLP networks have been evaluated by training them using the choosen set of features.

The whole process (features selection + network evaluation) can be executed using the script `main.m`.


## Authors

* **Paolo Sassi** - [paulrox](https://github.com/paulrox)
* **Matteo Rotundo** - [Gyro91](https://github.com/Gyro91)
