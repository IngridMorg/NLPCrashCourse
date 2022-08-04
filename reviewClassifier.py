#this is from lesson 7 of:
#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

#in this project i will use an 80% train and 20% test set approach
#with a 50/50 split in the test set between positive and negative reviews

#need to clean and tokenise the sets (we will save this to a file perhaps)

#i need to clean and tokenise all the data
import string

import numpy
from gensim.models import Word2Vec
from keras_preprocessing.sequence import pad_sequences
from matplotlib import pyplot
from numpy import array
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot

sentences = []
text = []
labels = array([])

#the following two functions are taken from wordEmbeddings.py
def loadText(file):
    # open file and read text into the variable 'text
    file = open(file, encoding="utf-8")

    text = file.read()
    file.close()
    # split the words based on newline characters
    text = text.splitlines()
    #used for the neural net
    global max_len
    max_len = 0
    for i in range(0,len(text)):
        text[i] = text[i].translate(str.maketrans('', '', string.punctuation))
        text[i] = text[i].lower()
        if (len(text[i]) > max_len):
            max_len = len(text[i])
    return text
def splitSentences(text, sentences):
    # 1 = Pos
    # 0 = Neg

    for t in text:
        sentences.append(t.split())

    #print(sentences)
    return sentences
#need to generate labels for the data
#load negative data
textN = loadText("data/Neg.txt")
sentences = splitSentences(textN, sentences)
length = len(sentences)
for i in range(0,len(sentences)):
    labels = numpy.append(labels, [0])
#load positive data
textP = loadText("data/Pos.txt")
sentences = splitSentences(textP, sentences)
length = len(sentences) - length
for i in range(0,length):
    labels = numpy.append(labels, [1])
#print(sentences)
#print(labels)
#print(len(labels))
#we now need to develop the embedding for the data
text = textN + textP
#print(text)
def embedd(sentences, freq):
    #we need to tokenise the words
    #t = kerasBag("data/damonText.txt")
    #train our model
    #print(t)
    model = Word2Vec(sentences, min_count=freq)
    #summarise the loaded model
    #print(model)
    #summarise the vocab
    words = list(model.wv.vocab)
    #print(words)
    #access the vector for a word
    #print(model['robots'])
    X = model[model.wv.vocab]
    #fit a 2d PCA model to the vectors
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    #create a scatter plot of the projection
    #embeddPlot(model, result)
    return model

def gensim_to_keras_embedding(model, train_embeddings=False):
    #https://github.com/RaRe-Technologies/gensim/wiki/Using-Gensim-Embeddings-with-Keras-and-Tensorflow
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    layer = Embedding(
        input_dim=weights.shape[0],
        output_dim=weights.shape[1],
        weights=[weights],
        trainable=train_embeddings,
    )
    return layer
def getVocabSize(sentences):
    model = Word2Vec(sentences, min_count=1)
    vocabSize = len(list(model.wv.vocab))
    return vocabSize
def embeddPlot(model, result):
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

print("embedding")
emModel = embedd(sentences, 10)
vocabSize = getVocabSize(sentences)
#print(vocabSize)

def encodeDocs(vocab_size, docs):
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    ##print(encoded_docs)
    return encoded_docs
print("encoding docs")
#this is a 2d array and i feel like it shouldnt be smdh - it's not dw
ed = encodeDocs(vocabSize * 13, text)
#print(ed)
def padDocs(encoded_docs, max_length):
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    #print(padded_docs)
    return padded_docs
print("padding docs")
pd = padDocs(ed,max_len)
#print(pd)

def createModel(vocab_size, max_length):
    # define model
    model = Sequential()
    model.add(Embedding(vocab_size*1000, 8, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    #model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
print("creating model")
m = createModel(vocabSize, max_len)

def fitModel(model, padded_docs, e, labels):
    #using a validation split lowers the accuracy but may well improve our actual application (yay!)
    model.fit(padded_docs, labels, epochs=e,verbose=0,validation_split=0.2)
    #evaluate
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('accuracy: %f' %(accuracy*100))

#print(pd.shape)
#print(labels.shape)
#print(vocabSize)
fitModel(m, pd, 1, labels)
