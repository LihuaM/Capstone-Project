from __future__ import division
import pandas as pd
import numpy as np
from feature_engineer import get_data, get_unigram_sentence, get_unigrams, split_data

def get_model_words(sentences):
    '''
    Get the names of the words in the model's vocabulary. Convert it to a set, for speed.
    INPUT: a list of question1 and question2
    OUTPUT: a set of unique words in question1 and question2
    '''
    model_words = set()
    for item in sentences:
        for i in item:
            model_words.add(i)
    return model_words

def make_feature_vec(df, model, model_words, num_features):
    '''
    Function to average all of the word vectors.
    INPUT: dataframe, model from which to get the feature vectors, set of all the model words,\
           number of features to use for word vector dimensionality
    OUTPUT: dataframe of vectors
    '''
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

def model_self_trained(sentences, num_features, min_word_count, num_workers, context, downsampling):
    '''
    Train a model based on the own dataset.
    INPUT: a list of question1 and question2, number of features to use for word vector dimensionality,\
           minimum word count, number of threads to run in parallel, number of threads to run in parallel,\
           context window size, downsample setting for frequent words
    OUTPUT:self_trained model
    '''
    model_self = Word2Vec(sentences, workers=num_workers, \
                         size=num_features, min_count=min_word_count, \
                         window=context, sample=downsampling)
    return model_self

def model_google_news(file_path, binary=True):
    '''
    Get the google_news model from given path.
    INPUT: filepath from which to get google_news model.
    OUTPUT: google_news model
    '''
    model_google = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)
    return model_google

if __name__ =='__main__':

    file_path = '../data/quora_duplicate_questions.tsv'
    # Get the data.
    df = get_data(file_path)
    # Get unigrams of the whole dataset.
    get_unigrams(df)
    # Combine question1 and question2 to list.
    sentences = df.question1_unigram.tolist() + df.question2_unigram.values.tolist()
    # Split the data into trainig datset(0.9) and testing dataset(0.1)
    df_train, df_test = split_data(df)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 1    # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model.
    model_self = Word2Vec(sentences, num_features, min_word_count, num_workers, context, downsampling)
    # Make the model much more memory-efficient.
    model_self.init_sims(replace=True)
    # Get the names of the words in the model's vocabulary
    model_words = get_model_words(sentences)
    # Get the training dataset vectors.
    df_train_vecs = make_feature_vec(df_train, model_self, num_features)
    # Get the testing dataset vectors.
    df_test_vecs = make_feature_vec(df_test, model_self, num_features)
    # Fit Random Forest Classifier model.
    rmc = RandomForestClassifier(n_estimators=10)
    rmc.fit(df_train_vecs, df_train_y)
    # Get the scores of testing dataset
    get_scores(rmc, df_test_vecs, df_test_y)


    # Get Google News pre_trainded model.
    model_google = model_google_news('/Users/lihuama/Downloads/GoogleNews-vectors-negative300.bin', binary=True )
    # Get names of the words in the Google News model's vocabulary.
    index2word_set = set(model_google_news.index2word)
    # Get training dataset vectors.
    df_train_vecs = make_feature_vec(df_train, model_google, index2word_set, num_features)
    # Get the testing dataset vectors.
    df_train_vecs = make_feature_vec(df_train, model_google,index2word_set, num_features)
    # Fit Random Forest Classifier model.
    rmc_modelg = RandomForestClassifier(n_estimators=10)
    rmc_modelg.fit(df_train_vecs, df_train_y)
    # Get the scores of testing dataset
    get_scores(rmc_modelg, df_test_vecs, df_test_y)
