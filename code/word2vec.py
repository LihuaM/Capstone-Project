from __future__ import division
import pandas as pd
import numpy as np
from feature_engineering import get_data, get_unigram_sentence, get_unigrams

def make_feature_vec(df, model, model_words, num_features):
    # Function to average all of the word vectors.
    counter = 0
    # Pre-initialize an empty numpy array (for speed)
    df_vecs = np.zeros((len(df), num_features), dtype= 'float32')
    for i in df.index.values:
        words = df.question1_unigram[i] + df.question2_unigram[i]
        set_words = set(words)
        nwords = 0.
        feature_vecs = np.zeros((num_features,), dtype= 'float32')
    #  Loop over each word in the question1 and question2 and, if it is in the model's vocaublary,\
    #  add its feature vector to the total.
        for word in set_words:
            if word in model_words:
                nwords = nwords + 1
                feature_vecs = np.add(feature_vecs,model[word])
        df_vecs[counter] = np.divide(feature_vecs, nwords + 1.)
        counter += 1
    return df_vecs



if __name__ =='__main__':

    file_path = '../data/quora_duplicate_questions.tsv'
    get_data(file_path)
    get_unigrams(df)
    sentences = df.question1_unigram.tolist() + df.question2_unigram.values.tolist()

    num_features = 300    # Word vector dimensionality
    min_word_count = 1    # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3

    model_self_trained = Word2Vec(sentences, workers=num_workers, \
                 size=num_features, min_count=min_word_count, \
                 window=context, sample=downsampling)

    model_words = set()
for item in sentences:
    for i in item:
        model_words.add(i)
