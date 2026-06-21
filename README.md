Project 6: Neural Network Memorization with Random Labels
Overview
This notebook demonstrates a key finding in deep learning: the ability of large neural networks to memorize completely random data while still achieving near-perfect training accuracy. This phenomenon challenges traditional statistical learning theory and highlights the unique generalization properties of overparameterized models.

The project uses an AlexNet-like convolutional neural network (AlexNetVariation) trained on a subset of the MNIST dataset (6,000 images) where the training labels have been completely randomized. For comparison, the model is also trained on the same data with its original, correct labels.

Model Architecture
The AlexNetVariation model is a convolutional neural network adapted for 28x28 grayscale input images (like MNIST). It consists of:

Convolutional Layers: Five convolutional layers with ReLU activations and Max Pooling, progressively extracting features.
Fully Connected Layers: Three fully connected layers with Dropout for regularization, leading to a 10-class output (for digits 0-9).
Total Parameters: 5,670,602

Key Findings
Memorization of Random Labels: Despite training on completely randomized labels, the neural network achieved near-perfect training accuracy (e.g., 90%+ in some runs, up to 100% with sufficient epochs and capacity). This demonstrates the network's capacity to 'memorize' noisy or meaningless data.
Poor Generalization with Random Labels: When trained on random labels, the test accuracy remained around 10% (equivalent to random chance), indicating that the network did not learn any meaningful patterns or features that could generalize to unseen data.
Contrast with Real Labels: Training on real, correct labels showed rapid convergence to high training and test accuracies, demonstrating the network's ability to learn generalizable features when presented with structured data.
Implications for Generalization Theory: This experiment supports the idea that neural networks can perfectly interpolate (memorize) training data without necessarily hurting test performance in certain regimes, leading to concepts like the "double descent" phenomenon.
Experiment Setup
Dataset: MNIST (10% subset = 6,000 images)
Labels: Completely randomized (Uniform from 1-10) for the main experiment, and real labels for comparison.
Optimization: SGD with learning rate 0.1 and momentum 0.9.
Epochs: 300 for random labels, 50 for real labels (due to faster convergence).
Batch Size: 64
Visualizations
The notebook generates plots showing:

Training accuracy and loss over epochs for the random label experiment.
A comparison of training accuracy and loss curves between models trained with random labels and real labels.
How to Run
Open the notebook in Google Colab.
Run all cells sequentially.
Observe the printed metrics and generated plots showcasing the memorization phenomenon.
References
Zhang et al. "Understanding Deep Learning Requires Rethinking Generalization" ICLR 2017.
