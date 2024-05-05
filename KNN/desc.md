
## Description
In this assignment, you will implement a kNN classifier to classify text documents using Python on top of Spark. You'll perform three main subtasks: data preparation, classification, and evaluation.

## Data
You'll use the “20 newsgroups” data set, which you previously used in Lab 5. This dataset consists of 19,997 posts from 20 different categories. The data can be accessed either via a direct [S3 link for same line format](https://s3.amazonaws.com/chrisjermainebucket/comp330_A6/20_news_same_line.txt).

## Tasks
### Task 1: Data Preparation
- Write Spark code to build a dictionary of the 20,000 most frequent words in the training corpus, ordering them by frequency.
- Create an RDD where each document is represented by a NumPy array, reflecting the frequency of each dictionary word in that document.
- For the arrays of documents `20 newsgroups/comp.graphics/37261`, `20 newsgroups/talk.politics.mideast/75944`, and `20 newsgroups/sci.med/58763`, print only the non-zero entries.

### Task 2: Classification
- Convert the count vectors from Task 1 into TF-IDF vectors. Print out the non-zero entries of the TF-IDF arrays for the same documents listed in Task 1.

### Task 3: Evaluation
- Implement the `predictLabel` function using the kNN algorithm to classify text strings based on their closest matching documents in the corpus.
- Test this function with provided excerpts from various Wikipedia articles.

