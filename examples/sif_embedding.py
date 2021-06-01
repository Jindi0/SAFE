import sys
sys.path.append('../src')
import data_io, params, SIF_embedding
import json
import numpy as np
import os

# input
wordfile = '../data/glove.840B.300d.txt' # word vector file, can be downloaded from GloVe website
weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
rmpc = 1 # number of principal components to remove in SIF weighting scheme

# set parameters
params = params.params()
params.rmpc = rmpc

keep_headline_words = 10  # the number of kept words in the headline
keep_body_workds = 100  # the number of kept words in the article body


# load word vectors and save as files
# (words, We) = data_io.getWordmap(wordfile)
# with open('../data/words.json', 'w') as f:
#     json.dump(works, f)

# np.save("../data/We.npy", We)


with open('../data/words.json', 'r') as f:
    words = json.load(f)

We = np.load('../data/We.npy')

# load word weights
word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word


def embedding(tokens):
    x, m = data_io.sentences2idx(tokens, words)
    w = data_io.seq2weight(x, m, weight4ind)
    result = SIF_embedding.SIF_embedding(We, x, w, params) 

    return result



# load news
datafolder = '../FakeNewsNet_Dataset_processed/'
newsfolders = os.listdir(datafolder)
for folder in newsfolders:
    folderpath = os.path.join(datafolder, folder)
    for news in os.listdir(folderpath):
        print(news)
        with open(os.path.join(datafolder, folder, news, 'cleaned.json'), 'r') as f:
            article = json.load(f)

        news_id = news.split('.')[0]
                                    
        headline_embedding = embedding(article['headline'][:keep_headline_words])
        np.save(os.path.join(datafolder, folder, news_id, 'headline.npy'), headline_embedding)

        body_embedding = embedding(article['body'][:keep_body_workds])
        np.save(os.path.join(datafolder, folder, news_id, 'body.npy'), body_embedding)
        


