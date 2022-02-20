import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

df_review = pd.read_csv('Musical_instruments_reviews.csv')
df_review.head()


## Copy the summary column data
df_summary = df_review[['summary']].copy()
df_summary.head()



## Tokenization
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import casual_tokenize

## Word tokenizer
def get_word_tokens():
    word_tokens = []
    for idx, row in df_summary.iterrows():
        for word in word_tokenize(row.summary):
            word_tokens.append(word)
    
    return word_tokens


## Sentence tokenizer
def get_sentence_tokens():
    sent_tokens = []
    for idx, row in df_summary.iterrows():
        for sentence in sent_tokenize(row.summary):
            sent_tokens.append(sentence)
            
    return sent_tokens



## Casual tokenizer
def get_casual_tokens():
    casual_tokens = []
    for idx, row in df_summary.iterrows():
        for cas_tokens in casual_tokenize(row.summary):
            casual_tokens.append(cas_tokens)
            
    return casual_tokens



## Stemming
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer

# Perform stemming
# Functions receives stemming type and perfrms 3 types of stemming
# PorterStemmer, LancasterStemmer and SnowballStemmer
def perform_stemming(tokenized_data, stemming_type):
    stemmed_data = []
    if stemming_type == 'PorterStemmer':
        for word in tokenized_data:
            stemmed_data.append(PorterStemmer().stem(word))
    elif stemming_type == 'LancasterStemmer':
        for word in tokenized_data:
            stemmed_data.append(LancasterStemmer().stem(word))
    elif stemming_type == 'SnowballStemmer':
        for word in tokenized_data:
            stemmed_data.append(SnowballStemmer('english').stem(word))
    else:
        print('Invalid stemming type received')
    
    return stemmed_data



## Lemmatization
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

# Lemmatization
def get_lemmatized_data(tokenized_data):
    lemmatizer = WordNetLemmatizer()
    
    lemmatized_data = []
    for word in tokenized_data:
        lemmatized_data.append(lemmatizer.lemmatize(word, 'v'))
    
    return lemmatized_data


if __name__ == "__main__":
    
    
    ## Tokenization
    # Word tokenization
    word_tokens = get_word_tokens()
    print('\n---------------- Tokenization --------------- ')
    print('PASS: Word tokenization')
    
    # Sentence tokeniztion
    sent_tokens = get_sentence_tokens()
    print('PASS: Sentence tokenization')
    
    # Casual tokenization
    casual_tokens = get_casual_tokens()
    print('PASS: Casusl tokenization')
    
    print('\n---------------- Stemming --------------- ')
    ## Stemming
    # PorterStemmer  
    stemmed_tokens = perform_stemming(word_tokens, 'PorterStemmer')
    porter_stemmer_result_data = dict(zip(word_tokens, stemmed_tokens))
    print('PASS: Porter stemming')

    # LancasterStemmer
    stemmed_tokens = perform_stemming(word_tokens, 'LancasterStemmer')
    lancaster_stemmer_result_data = dict(zip(word_tokens, stemmed_tokens))
    print('PASS: Lancaster stemming')

    # SnowballStemmer
    stemmed_tokens = perform_stemming(word_tokens, 'SnowballStemmer')
    snowball_stemmer_result_data = dict(zip(word_tokens, stemmed_tokens))
    print('PASS: Snowball stemming')

    print('\n---------------- Lemmatization --------------- ')
    ## Lemmatization
    lemmatized_data = get_lemmatized_data(word_tokens)
    lemma_result_data = dict(zip(word_tokens, lemmatized_data))
    print('PASS: Lemmatized data')
    
    print('\n---------------------------------------------- ')
    print('Successully parsed summary data using :: Tokenization, Stemming, Lemmatization')
    print('\n---------------------------------------------- ')
    print('\nFinished Run')
