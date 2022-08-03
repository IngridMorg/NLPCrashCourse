#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

#word embeddings = word representation that allows words with simlar meaning to have a similar representation
#word embedding s learn a real-valued vector representation for a predefined fixed size voabulary from a corpus of text

#word embeddings can be trained
from gensim.models import Word2Vec
from bagOfWords import *
#word2vec my beloved

text = loadText("data/damonText.txt")

def embedd(text):
    #train our model
    model = Word2Vec(text, min_count=1)
    #summarise the loaded model
    print(model)
    #summarise the vocab
    #words = list(model.wv)
    #print(words)
    #access the vector for a word
    print(model['robots'])


print("wordEmbeddings")
embedd(text)
print("end of word embeddings")