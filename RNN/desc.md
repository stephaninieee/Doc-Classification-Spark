
## Description
In this assignment, you will utilize Google's TensorFlow to implement deep learning architectures aimed at classifying sequences of raw text based on characters. Given the computational demands of deep learning, using a GPU is highly recommended. You can choose between running your code on Amazon EC2 or Google Colab, depending on your resource availability and cost considerations.

## Prerequisites
- Python
- TensorFlow
- Access to Amazon EC2 or Google Colab


## Setup Instructions

### Amazon EC2
1. **Instance Setup**: Log into Amazon AWS, navigate to EC2 service and launch an instance using the "Deep Learning AMI (Ubuntu 18.04) Version 62.0".
2. **Choose Instance Type**: Select `g3s.xlarge` for adequate GPU support.
3. **Accessing the Instance**: Use SSH to access your instance with `ubuntu` as the username.
4. **Environment Preparation**: Activate the TensorFlow environment using `source activate tensorflow2_p38` and start Python with `python`.

### Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com).
2. Start a new notebook and ensure that a GPU is enabled via "Change runtime type" selecting "GPU".

## Data
The dataset consists of text from three files, which will be used for training and testing the models:
- [Holmes.txt](https://s3.amazonaws.com/chrisjermainebucket/text/Holmes.txt)
- [War.txt](https://s3.amazonaws.com/chrisjermainebucket/text/war.txt)
- [William.txt](https://s3.amazonaws.com/chrisjermainebucket/text/william.txt)

## Tasks 

### Task 0: Running RNN Learning Using TensorFlow
- Initialize the TensorFlow environment and run the provided RNN code.
- Load the provided texts and execute the RNN model for 10,000 iterations. Document the output of the last 20 iterations.

### Task 1: Modifying the RNN Code to Compute Accuracy on a Test Set
- Modify the code to separate training and test data ensuring no overlap. After training, output the loss and accuracy for 3000 randomly selected documents from the test set.

### Task 2: Adding "Time Warping" to the RNN
- Implement time warping in the RNN to improve handling of vanishing gradients.
-  Adjust the RNN architecture to include state information from both the previous and ten time ticks prior. Modify the hidden layer size accordingly and document the performance.

### Task 3: Implementing a Feed-Forward Network
-  Replace the RNN with a feed-forward network.
- Change the input data handling from sequences of vectors to a single concatenated vector per text line. Experiment with different network configurations and layer sizes to optimize test accuracy.

### Task 4: Modifying the "Time Warping" RNN to Use a Convolution
- Incorporate convolutional filters into the RNN.
- Process sequences of characters through multiple filters to capture patterns effectively. Experiment with the length of character windows and the number of filters to enhance accuracy.


