#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

#word embeddings = word representation that allows words with simlar meaning to have a similar representation
#word embedding s learn a real-valued vector representation for a predefined fixed size voabulary from a corpus of text

#word embeddings can be trained
from gensim.models import Word2Vec
from bagOfWords import *
from cleanData import clean
from sklearn.decomposition import PCA
from matplotlib import pyplot
#word2vec my beloved
#load the text into line by line
def loadText(file):
    # open file and read text into the variable 'text
    file = open(file, encoding="utf-8")

    text = file.read()
    file.close()
    # split the words based on newline characters
    text = text.splitlines()
    for i in range(0,len(text)):
        text[i] = text[i].translate(str.maketrans('', '', string.punctuation))
        text[i].lower()
    return text
#now a function to split each line into words
#we want a list of lists (aka a generator of strings (see bagOfOWords.py))
def splitSentences(text):
    sentences = []
    for t in text:
        sentences.append(t.split())
    #print(sentences)
    return sentences
def embedd(sentences):
    #we need to tokenise the words
    #t = kerasBag("data/damonText.txt")
    #train our model
    #print(t)
    model = Word2Vec(sentences, min_count=1)
    #summarise the loaded model
    #print(model)
    #summarise the vocab
    words = list(model.wv.vocab)
    print(words)
    #access the vector for a word
    #print(model['robots'])
    X = model[model.wv.vocab]
    #fit a 2d PCA model to the vectors
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    #create a scatter plot of the projection
    pyplot.scatter(result[:,0],result[:,1])
    words = list(model.wv.vocab)
    for i,word in enumerate(words):
        pyplot.annotate(word, xy=(result[i,0],result[i,1]))
    pyplot.show()

print("wordEmbeddings")
#embedd(text)
text = loadText("data/damonText.txt")
sentences = splitSentences(text)
embedd(sentences)
print("end of word embeddings")