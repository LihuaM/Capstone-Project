# Capstone-Project
## Description

Semantic similarity is basically deciding how similar two documents are to each other, and assessing it is quite useful for things like identifying duplicate posts, semi-supervised labelling, whether two news articles are talking about the same thing, and lots of other applications.

In order to build a high-quality knowledge base, it's important to ensure each unique question exists on Quora only once. Writers shouldn't have to write the same answer to multiple versions of the same question, and readers should be able to find a single canonical page with the question they're looking for. For example, we'd consider questions like “What are the best ways to lose weight?”, “How can a person reduce weight?”, and “What are effective weight loss plans?” to be duplicate questions because they all have the same intent. To prevent duplicate questions from existing on Quora, we need to develop machine learning and natural language processing systems to automatically identify when questions with the same intent have been asked multiple times.

## Problem Definition

More formally, the duplicate detection problem can be defined as follows: given a pair of questions q1 and q2, train a model that learns the function:

 f(q1, q2) → 0 or 1

 where 1 represents that q1 and q2 have the same intent and 0 otherwise.

## Two methods to solve the problems
* Method One -- Regular Feature Engineering
  I kept four engineered features in the final model, which were common_unigram_ratio, common_bigram_ratio, common_trigram_ratio, cosine_similarity. I tried Random Forest Classifier and Gradient Boosting Classifier
  to fit my model.

* Method Two -- Word2Vec
  I used two models, one was self_trained model based on my own dataset, the other was google_news model. I used 300 vector features for the dataset and fit the Random Forest Classifier to both models.

## Result
   The best result was achieved by using Word2Vec model based on self_trained model. The scores were:
   
   Accuracy Score | Recall Score | Precision Score | AUC Score
   -------------  | -------------| ----------------|------------
       0.77       |     0.55     |       0.76      |    0.83


## Data Source

Data was from Quora. 404290 Paired questions with 1 and 0 labels indicating they are the same questions or not.
