# Deep Learning
## Project: Plagiarism Detection

## Project Overview
In this project, you will be tasked with building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious. You'll be defining a few different similarity features, as outlined in [this paper](https://s3.amazonaws.com/video.udacity-data.com/topher/2019/January/5c412841_developing-a-corpus-of-plagiarised-short-answers/developing-a-corpus-of-plagiarised-short-answers.pdf), which should help you build a robust plagiarism detector!

This project will be broken down into three main notebooks:

**Notebook 1: Data Exploration**
* Load in the corpus of plagiarism text data.
* Explore the existing data features and the data distribution.
* This first notebook is **not** required in your final project submission.

**Notebook 2: Feature Engineering**
* Clean and pre-process the text data.
* Define features for comparing the similarity of an answer text and a source text, and extract similarity features.
* Select "good" features, by analyzing the correlations between different features.
* Create train/test `.csv` files that hold the relevant features and class labels for train/test data points.

**Notebook 3: Train and Deploy Your Model in SageMaker**
* Upload your train/test feature data to S3.
* Define a binary classification model and a training script.
* Train your model and deploy it using SageMaker.
* Evaluate your deployed classifier.

## Project Highlights

###Type of Plagiarism
- Dataset contains multiple text file. Each text file is associated with one Task (task A-E) and one Category of plagiarism, which you can see in the above DataFrame.

**Tasks, A-E**
- Each text file contains an answer to one short question; these questions are labeled as tasks A-E. For example, Task A asks the question: "What is inheritance in object oriented programming?"

**Categories of plagiarism**
Each text file has an associated plagiarism label/category:

1. Plagiarized categories: cut, light, and heavy.
   - These categories represent different levels of plagiarized answer texts. cut answers copy directly from a source text, light answers are based on the source text but include some light rephrasing, and heavy answers are based on the source text, but heavily rephrased (and will likely be the most challenging kind of plagiarism to detect).
2. Non-plagiarized category: non.
   - non indicates that an answer is not plagiarized; the Wikipedia source text is not used to create this answer.
3. Special, source text category: orig.
   - This is a specific category for the original, Wikipedia source text. We will use these files only for comparison purposes.

### Feature Engineering / Similarity Features

**Containment**
Your first task will be to create containment features. To understand containment, let's first revisit a definition of [n-grams](https://en.wikipedia.org/wiki/N-gram). An n-gram is a sequential word grouping. For example, in a line like "bayes rule gives us a way to combine prior knowledge with new information," a 1-gram is just one word, like "bayes." A 2-gram might be "bayes rule" and a 3-gram might be "combine prior knowledge."

Containment is defined as the intersection of the n-gram word count of the Wikipedia Source Text (S) with the n-gram word count of the Student Answer Text (S) divided by the n-gram word count of the Student Answer Text.

If the two texts have no n-grams in common, the containment will be 0, but if all their n-grams intersect then the containment will be 1. Intuitively, you can see how having longer n-gram's in common, might be an indication of cut-and-paste plagiarism. In this project, it will be up to you to decide on the appropriate n or several n's to use in your final model.

**Longest Common Subsequence**
Containment a good way to find overlap in word usage between two documents; it may help identify cases of cut-and-paste as well as paraphrased levels of plagiarism. Since plagiarism is a fairly complex task with varying levels, it's often useful to include other measures of similarity. The paper also discusses a feature called longest common subsequence.

The longest common subsequence is the longest string of words (or letters) that are the same between the Wikipedia Source Text (S) and the Student Answer Text (A). This value is also normalized by dividing by the total number of words (or letters) in the Student Answer Text.

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [Torch](https://pytorch.org/)
- [Sagemaker](https://sagemaker.readthedocs.io/en/stable/)
- [Boto3](https://boto3.readthedocs.io/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Code

This project contains three files:

- `1_Data_Exploration.ipynb`: Jupyter Notebook for Data Exploration.
- `2_Plagiarism_Feature_Engineering.ipynb`: Jupyter Notebook for Feature Engineering.
- `3_Training_a_Model.ipynb`: Jupyter Notebook for training the model.
- `model.py`: A Python file that is used to construct the model.
- `train.py`: A Python file to train the model.
- `predict.py`: A Python file that contains custom inference code. 
- `helpers.py`: A Python file containing helper functions `create_text_column` and `train_test_dataframe`.
- `problem_unittests.py`: A Python file containing unit test code that is run behind-the-scenes. Do not modify

### Run

In a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
ipython notebook 1_Data_Exploration.ipynb
ipython notebook 2_Plagiarism_Feature_Engineering.ipynb
ipython notebook 3_Training_a_Model.ipynb
```  
or
```bash
jupyter notebook 1_Data_Exploration.ipynb
jupyter notebook 2_Plagiarism_Feature_Engineering.ipynb
jupyter notebook 3_Training_a_Model.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

Data used for this project is a slightly modified version of a dataset created by Paul Clough (Information Studies) and Mark Stevenson (Computer Science), at the University of Sheffield. You can read all about the data collection and corpus, at [their university webpage](https://ir.shef.ac.uk/cloughie/resources/plagiarism_corpus.html).
