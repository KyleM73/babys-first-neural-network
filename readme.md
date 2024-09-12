# Baby's First Neural Network
In this repository we will explore simple neural networks, starting with a multi-layer perceptron (MLP). We will derive both the forward and backward pass of a neural network and train it using (stochastic) gradient descent (SGD). 

Suggested usage of this repository:
- Notation and math preliminaries can be found in the [notation](notation.md) notes.
- Read the background [notes](notes.md) on MLPs and fill in the blanks to derive the forward and backward passes and weight updates.
- Fill in each segment of code marked ``TODO`` in each file in the ``source`` folder. You may run the [train.py](source/train.py) script to test your implementation. 
- Run the [test.py](source/test.py) script to run a suite of test cases against a reference implementation. 
- Note that the reference implementation can be found in the ``source/test`` folder, but it is advised that you attempt the derivations without looking at it.

## Setup
### Requirements
- python
- numpy

We assume that you have a python development environment with numpy installed. If you want to work inside a conda environment, the environment can be created with ``conda create -n nn python=3.12`` and activated via ``conda activate nn``. Then numpy can be installed via ``pip install numpy``. The specific version of python3.x does not matter.