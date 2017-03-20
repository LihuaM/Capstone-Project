# Capstone-Project
## Description

In order to build a high-quality knowledge base, it's important to ensure each unique question exists on Quora only once. Writers shouldn't have to write the same answer to multiple versions of the same question, and readers should be able to find a single canonical page with the question they're looking for. For example, we'd consider questions like “What are the best ways to lose weight?”, “How can a person reduce weight?”, and “What are effective weight loss plans?” to be duplicate questions because they all have the same intent. To prevent duplicate questions from existing on Quora, we need to develop machine learning and natural language processing systems to automatically identify when questions with the same intent have been asked multiple times.

## Problem Definition

More formally, the duplicate detection problem can be defined as follows: given a pair of questions q1 and q2, train a model that learns the function:

 f(q1, q2) → 0 or 1

 where 1 represents that q1 and q2 have the same intent and 0 otherwise.

## Possible approaches to solve the problems

 I want to try Random Forest with different engineered features. For example, cosine similarity of the tokens, the number of common words, the number of common topics labeled on the questions, and then if possible, use Recurrent Neural Networks (RNNs) to build the model to solve the problems.

## Data Source

Data was from Quora. 404290 Paired questions with 1 and 0 labels indicating they are the same questions or not.
