###############################################################################
# The code works on the raw dataset FakeNewsNet_Dataset
# 1. clean useless files and the instances with empty article headline or body.
# 2. preprocess the data and store the result in FakeNewsNet_Dataset_processed
###############################################################################
import os
import json
import shutil
from string import punctuation, digits
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
stop_words = stopwords.words("english")


# clean useless files and empty instances
def clean_files(datafolder):
    newsfolders = os.listdir(datafolder)
    for folder in newsfolders:
        folderpath = os.path.join(datafolder, folder)
        for news in os.listdir(folderpath):
            os.remove(os.path.join(folderpath, news, 'likes.json')) 
            os.remove(os.path.join(folderpath, news, 'replies.json')) 
            os.remove(os.path.join(folderpath, news, 'retweets.json')) 
            os.remove(os.path.join(folderpath, news, 'tweets.json')) 
            with open(os.path.join(folderpath, news, 'news_article.json'), 'r') as f:
                article = json.load(f)
            if len(article) == 0 or len(article['title']) == 0 or len(article['text']) == 0:
                shutil.rmtree(os.path.join(folderpath, news)) 


    
# clean data
def clean_article(datafolder, datafolder_clean):
    newsfolders = os.listdir(datafolder)
    for folder in newsfolders:
        if not os.path.exists(os.path.join(datafolder_clean, folder) ):
            os.mkdir(os.path.join(datafolder_clean, folder))
        folderpath = os.path.join(datafolder, folder)
        for news in os.listdir(folderpath):
            clean_tokens = {}
            with open(os.path.join(folderpath, news, 'news_article.json'), 'r') as f:
                article = json.load(f)
            clean_tokens['headline'] = clean_text(article['title'])
            clean_tokens['body'] = clean_text(article['text'])

            if len(clean_tokens['headline']) > 0 and len(clean_tokens['body']) > 0:

                if not os.path.exists(os.path.join(datafolder_clean, folder, news)):
                    os.mkdir(os.path.join(datafolder_clean, folder, news))

                with open(os.path.join(datafolder_clean, folder, news, 'cleaned.json'), 'w') as f:
                    json.dump(clean_tokens, f)
            print(news)



# text preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    text = re.sub(remove_chars, ' ', text)

    text = text.strip()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]

    stemmer= PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    lemmatizer=WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens




if __name__ == '__main__':
    datafolder = '../FakeNewsNet_Dataset/'

    clean_files(datafolder)   # clean useless files

    datafolder_clean = '../FakeNewsNet_Dataset_processed/'
    if not os.path.exists(datafolder_clean):
        os.mkdir(datafolder_clean)
    clean_article(datafolder, datafolder_clean)   # preprocesse text


    
    