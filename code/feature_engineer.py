from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import ngrams
from gensim.models import Word2Vec
import logging

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', 100)


def get_data(file_path):
    '''
    INPUT: file path from which to import the data
    OUTPUT: dataframe
    '''
    df = pd.read_table(file_path)
    df.dropna(inplace = True)
    df.drop(['id','qid1','qid2'], axis=1, inplace = True)
    return df

def duplicate_or_not_plot(df):
    '''
    INPUT: dataframe
    OUTPUT: plot to show the count of Duplicated or Not_Duplicated questions.
    '''
    plt.rcParams['figure.figsize'] = (8, 6)
    df.groupby('is_duplicate').is_duplicate.count().plot(kind='bar', rot=0)
    plt.xlabel('Duplicate or Not')
    plt.ylabel('Count of Paired Questions')
    plt.title('Count of Duplicated or Not_Duplicated Questions', fontsize=15)
    plt.show()

def split_data(df):
    '''
    Split the data into training dataset(0.9) and testing dataset(0.1)
    INPUT: dataframe
    OUTPUT: training dataset and testing dataset
    '''
    X = df[['question1', 'question2']]
    y = df.is_duplicate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    return df_train, df_test

def get_unigram_sentence(sentence):
    '''
    INPUT: sentence from which to get unigrams
    OUTPUT: a list of unigrams of the sentence
    '''
    return [word for word in word_tokenize(sentence.lower()) if word not in stopwords_set and\
            word not in punctuation]

def get_unigrams(df):
    '''
    INPUT: dataframe
    OUTPUT: NONE
    '''
    df['question1_unigram'] = df['question1'].apply(lambda x: get_unigram_sentence(x.decode(encoding='utf-8')))
    df['question2_unigram'] = df['question2'].apply(lambda x: get_unigram_sentence(x.decode(encoding='utf-8')))

def get_common_unigram_ratio(df):
    '''
    INPUT: dataframe
    OUTPUT: NONE
    '''
    df['common_unigram_count'] = df.apply(lambda x: len(set(x['question1_unigram']).intersection\
                                 (set(x['question2_unigram']))), axis=1)
    df['unigram_count'] = df.apply(lambda x: max(len(set(x['question1_unigram']).union\
                          (set(x['question2_unigram']))), 1), axis=1)
    df['common_unigram_ratio'] = df['common_unigram_count'] / df'unigram_count']

def common_unigram_ratio_plot(df):
    '''
    INPUT: dataframe
    OUTPUT: plot to show common_unigram_ratio diffrence of two classes
    '''
    plt.figure(figsize=(8,6))
    sns.violinplot(x='is_duplicate', y='common_unigram_ratio', data=df)
    plt.title('Common_Unigram_Ratio Difference Between Two Classes', fontsize=20)
    plt.savefig('image/common_unigram_ratio_violinplot')

def get_bigrams(df):
    '''
    Get bigrams of question1 and question2.
    INPUT: dataframe
    OUTPUT: None
    '''
    df['question1_bigram'] = df['question1_unigram'].apply(lambda x: [i for i in ngrams(x, 2)])
    df['question2_bigram'] = df['question2_unigram'].apply(lambda x: [i for i in ngrams(x, 2)])
#
def get_common_bigram_ratio(df):
    '''
    Get common_bigram_ratio of question1 and question2.
    INPUT: dataframe
    OUTPUT: NONE
    '''
    df['common_bigram_count'] = df.apply(lambda x: len(set(x['question1_bigram']).intersection\
                                                               (set(x['question2_bigram']))), axis=1)
    df['bigram_count'] = df.apply(lambda x: max(len(set(x['question1_bigram']).union\
                                                            (set(x['question2_bigram']))), 1), axis=1)
    df['common_bigram_ratio'] = df['common_bigram_count'] / df['bigram_count']

def common_bigram_ratio_plot(df):
    '''
    INPUT: dataframe
    OUTPUT: plot to show common_bigram_ratio diffrence of two classes
    '''
    plt.figure(figsize=(8,6))
    sns.violinplot(x='is_duplicate', y='common_bigram_ratio', data=df)
    plt.title('Common_Bigram_Ratio Difference Between Two Classes', fontsize=20)
    plt.savefig('image/common_bigram_ratio_violinplot')

def get_trigrams(df):
    '''
    Get trigrams of question1 and question2.
    INPUT: dataframe
    OUTPUT: NONE
    '''
    df['question1_trigram'] = df['question1_unigram'].apply(lambda x: [i for i in ngrams(x, 3)])
    df['question2_trigram'] = df['question2_unigram'].apply(lambda x: [i for i in ngrams(x, 3)])

def get_common_trigram_ratio(df):
    '''
    Get common_trigram_ratio of question1 and question2.
    INPUT: dataframe
    OUTPUT: NONE
    '''
    df['common_trigram_count'] = df.apply(lambda x: len(set(x['question1_trigram']).intersection\
                                                                (set(x['question2_trigram']))), axis=1)
    df['trigram_count'] = df.apply(lambda x: max(len(set(x['question1_trigram']).union\
    (set(x['question2_trigram']))), 1), axis=1)
    df['common_trigram_ratio'] = df['common_trigram_count'] / df['trigram_count']

def common_trigram_ratio_plot(df):
    '''
    INPUT: dataframe
    OUTPUT: plot to show common_trigram_ratio diffrence of two classes
    '''
    plt.figure(figsize=(8,6))
    sns.violinplot(x='is_duplicate', y='common_trigram_ratio', data=df)
    plt.title('Common_Trigram_Ratio Difference Between Two Classes', fontsize=20)
    plt.savefig('image/common_trigram_ratio_violinplot')

def get_cosine_similarity(df):
    '''
    Get cosine similarity of question1 and question2
    INPUT: dataframe
    OUTPUT: NONE
    '''
    tfidf_matrix_transform1 = tfidf_vectorizer.transform(df.question1.values)
    tfidf_matrix_transform2 = tfidf_vectorizer.transform(df.question2.values)
    cos = []
    for i in range(len(df_train)):
        cos.append(cosine_similarity(tfidf_matrix_transform1[i].toarray(), tfidf_matrix_transform2[i].toarray()))
    df['cosine_similarity'] = cos
    df['cosine_similarity'] = df.cosine_similarity.apply(lambda x: x[0][0])

def cosine_similarity_plot(df):
    '''
    INPUT: dataframe
    OUTPUT: plot to show cosine_similarity diffrence of two classes
    '''
    sns.violinplot(x='is_duplicate', y='cosine_similarity', data=df)
    plt.title('Cosine_Similarity Difference Between Two Classes', fontsize=15)
    plt.savefig('cosine_similarity_violinplot')

def get_features(df):
    '''
    Get all the engineered features of df
    INPUT: dataframe
    OUTPUT: NONE
    '''
    get_unigrams(df_train)
    get_common_unigram_ratio(df_train)
    get_bigrams(df_train)
    get_common_bigram_ratio(df_train)
    get_trigrams(df_train)
    get_common_trigram_ratio(df_train)
    get_cosine_similarity(df_train)

def get_

def build_model(model, df_X, df_y):
    model.fit(df_X, df_y)
    return model

def get_scores(model, df_test_X, df_test_y):
    model_accuracy_score = accuracy_score(df_test_y, model.predict(df_test_X))
    model_recall_score = recall_score(df_test_y, model.predict(df_test_X))
    model_precision_score = precision_score(df_test_y, model.predict(df_test_X))
    model_roc_auc_score = roc_auc_score(df_test_y, model.predict_proba(df_test_X)[:,1])
    print 'Accuracy Score = {:.2f}\nRecall Score= {:.2f}\nPrecision Score = {:.2f}\nAUC_Score = {:.2f}'.\
      format(model_accuracy_score, model_recall_score, model_precision_score, model_roc_auc_score))

if __name__ == '__main__':
    file_path = '../data/quora_duplicate_questions.tsv'
    df = get_data(file_path)
    #duplicate_or_not_plot(df)
    df_train, df_test = split_data(df)
    stopwords_set = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    tfidf_vectorizer = TfidfVectorizer(stop_words=stopwords_set)
    tfidf_matrix = tfidf_vectorizer.fit(df_train.question1.values + df_train.question1.values)
    get_featues(df_train)

    feature_cols = ['common_unigram_ratio', 'common_bigram_ratio','common_trigram_ratio',\
                'cosine_similarity']
    df_train_X = df_train[feature_cols]
    df_train_y = df_train.is_duplicate

    model = GradientBoosingClassifer(leaning_rate=0.1, max_features='sqrt', n_estimators=300)
    buil_model(model, df_train_X, df_train_y)

    get_features(df_test)
    df_test_X = df_test[feature_cols]
    df_test_y = df_test.is_duplicate
    get_scores(model, df_test_X, df_test_y)
