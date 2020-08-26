# Deep-learning-for-classification-and-regression
Data loading and deep model training for classification and regression tasks on MNIST, Fashion-MNIST and LSP datasets 

1. Data Loading for Classification:

MNIST and Fashion-MNIST datasets are used. These datasets are loaded in two different ways: using the torchvision.datasets subclasses and using a custom dataset loader. In the first case, it checks whether the dataset is locally stored and download it from the web if it is not locally stored. Finally, the data samples and labels will be visualized.

2. Data Loading for Regression:

Leeds Sports Pose (LSP) dataset is used. The data will be loaded with a custom data loader. This is a dataset for 2D human pose estimation where the input is an image and the output is the human body pose, represented by a set of keypoints. A keypoint is a 2D coordinate. To demonstrate the data loading, the images and the corresponding labels will be visualized. The labels in this dataset are keypoints. For each image, the body skeleton will be drawn.

3. Data loading for regression with LSP dataset using a custom data loader with variable batch size and splitting dataset to train and test sets

4. Model Training for Classification

The Fashion-MNIST dataset is employed for learning a classifier. The classifier will be a convolutional neural network. It includes 2 nn.Conv2d operations and 3 nn.Linear operations. The activation is ReLU. The loss function is the cross-entropy. Training is implemented based on the train set of the dataset. The test set will be used only for evaluation. The train and test sets are separated in the dataset loading section.

The parameters of the network are learned based on the computation of the gradients. The optimization is based on Pytorch (torch.optim).

Two different plots are drawn after the model training. The error per epoch for both the train and test sets, and the accuracy per epoch for both the train and test sets.

5. Model Training for Regression

The LSP dataset is employed for learning a regressor. The custom dataloader which has been constructed for LSP is used here. The regressor will be a convolutional neural network. The architecture includes 3 nn.Conv2d operations and 4 nn.Linear operations. The final activation has to be linear and the loss function will be the mean squared error.

The parameters of the network are learned based on the computation of the gradients and the optimization is based on Pytorch (torch.optim).

Finally after the model training, the loss per epoch for both the train and test sets is plotted.
