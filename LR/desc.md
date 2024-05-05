
## Description
In this assignment, you will implement a regularized logistic regression model to classify text documents as either Wikipedia pages or Australian court cases. The implementation will be done in Python, on top of Spark, and will require the use of Amazon AWS to handle large datasets.

## Getting Started

### Prerequisites
- Python
- Apache Spark
- Access to Amazon AWS

### Data
Three datasets are provided:
1. **Training Data Set**: [TrainingDataOneLinePerDoc.txt](https://s3.amazonaws.com/chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt) (1.9 GB)
2. **Testing Data Set**: [TestingDataOneLinePerDoc.txt](https://s3.amazonaws.com/chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt) (200 MB)
3. **Small Data Set**: [SmallTrainingDataOneLinePerDoc.txt](https://s3.amazonaws.com/chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt) (37.5 MB)


## Tasks
### Task 1: Data Preparation
Build a dictionary of the 20,000 most frequent words from the training corpus. Validate by providing the frequency positions of specific words.

### Task 2: Learning
Use TF-IDF vectors and gradient descent to learn the logistic regression model. Implement L2 regularization and derive your own gradient descent formula based on classroom materials.

### Task 3: Evaluation
Evaluate the model using the testing dataset and compute the F1 score. Analyze false positives to understand classification errors.


### Task 2: Learning
Train a logistic regression model using TF-IDF vectors and gradient descent with L2 regularization. 
- Convert document counts into TF-IDF vectors using Spark.
- Implement a gradient descent algorithm to adjust weights of the logistic model:
    - Start with the LLH (Log-Likelihood) function discussed in class.
    - Derive the gradient update formula specifically for logistic regression.
    - Implement L2 regularization to prevent overfitting.
- At the end of learning, identify the fifty words with the largest positive weights in the model.
- Output the Model parameters and a list of significant words.


### Task 3: Evaluation
- Assess the performance of the logistic regression model using the testing dataset.
- Apply the trained model to the testing data to predict classifications.
- Calculate the F1 score to evaluate model performance.
- Analyze any false positives to understand potential classification errors:
- Examine the content of mistaken documents.
- Discuss possible reasons for misclassification, such as specific language or contextual similarities with Australian court cases.
 
