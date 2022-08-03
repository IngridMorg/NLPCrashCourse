#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/
#supplementary knowledge from:
#https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
#keras provides a way to use an embedding layer to learn a word embedding distrubuted representation for words as a part
#of fitting a deep learning model

#the keras embedding layer requires input data to be integer encoded (each word is a unique integer)
    #this can be done with tokeniser
    #input_dim = vocabulary size
    #output_dim = the size of the vector space of the embedding
    #input_length = number of words in input sequences
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import one_hot
from numpy import array


def compileModel(vocab_size, max_length):

    #define the model
    model = Sequential()
    #8 = the dimension (output_dim) which is the size of the vector space of the embedding
    #we can also initialise the embedding layer with pretrained weights like the ones produced with gensim
    model.add(Embedding(vocab_size, 8, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1,activation='sigmoid'))

    #compile the model
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    #summarise the model
    print(model.summary())
    return model

#we are gonna steal data from the tutorial
docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']
# define class labels (telling us pos or neg)
labels = array([1,1,1,1,1,0,0,0,0,0])

def encodeDocs(vocab_size, docs):
    encoded_docs = [one_hot(d, vocab_size) for d in docs]
    ##print(encoded_docs)
    return encoded_docs
#we now need to pad the docs for the maximum length we have found
def padDocs(encoded_docs, max_length):
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)
    return padded_docs

ed = encodeDocs(50, docs)
pd = padDocs(ed, 4)
m = compileModel(50, 4)

def fitModel(model, padded_docs, e, labels):
    model.fit(padded_docs, labels, epochs=e,verbose=0)
    #evaluate
    loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
    print('accuracy: %f' %(accuracy*100))

fitModel(m, pd, 50, labels)