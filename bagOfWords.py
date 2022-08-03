#lesson 3
#bag-of-words encoding using scikit and python
from sklearn.feature_extraction.text import TfidfVectorizer
#we need to load and tokenise the text
from cleanDataManual import *
from keras.preprocessing.text import Tokenizer
def loadText(file):
    # open file and read text into the variable 'text
    file = open(file, encoding="utf-8")

    text = file.read()
    file.close()
    # split the words based on newline characters
    text = text.splitlines()
    for i in range(0,len(text)):
        text[i] = text[i].translate(str.maketrans('', '', string.punctuation))
    return text

def scikitBag(file):
    #tfidf helps show us how relevant certain words are to a given document
    #and can be a great way to weight words in a document
    #an encoder can be created and trained on a corpus, then can be used multiple times
    #on any new data
    #an encode to score words based on the count is called CountVEctorizer
    #HashingVectorizer also is a hash function that reduces the vector length of each word
    #Tfid uses a score based on the word ocurrence and the inverse ocurrence
    #tfidf vectorizer transforms text to FEATURE VECTORS tgat
    #this function will use tfidVectorizer
    text = loadText(file)

    vectorizer = TfidfVectorizer()
    #tokenise and build the vocabulary
    vectorizer.fit(text)
    #summarise the vocab
    print(vectorizer.vocabulary_)
    print(vectorizer.idf_)
    #now we encode the document
    vector = vectorizer.transform([text[0]])
    #summarise the encoded vector
    print(vector.shape)
    print(vector.toarray())
    print(vector)
    return vector



def kerasBag(file):
    #in keras bag of words is used through the Tokenizer object
    #tokenizer allows us to vectorise a text corpus bu turning each text into either a
    #sequence of integers or into a vector(where the coefficient for each token can be binary, based on ocurrence or tf-idf etc
    #by defaul all punctuation is removed, but words may contain '
        #these sequences are then split into lists of tokens
        #0 is a reserved index that wont be assigned to any word
    # a sequence is a list of integer word indices
    # a text can be a list of strings, generator of strings or a list of lists of strings
    #we can also get the configuration that the tokenizer is using (get_config) which returns a python dictionary with
    #the tokenizers confuguration
    #sequences_to_matrix converts a list of sequences (integer word indices) to a numpy matrix
    #sequences_to_texts, transforms each sequence to a list of text
        #only the top (num_of_words -1) will be taken into account
        #sequences_to_texts generator, transforms each sequence in a list of sequences to a list of texts(strings)
    #texts_to_matrix, converts a list of texts to a numpy matrix

    text = loadText(file)

    t = Tokenizer()
    #fit tokeizer to documents
    t.fit_on_texts(text)
    #summarise
    #this is  a dictionary of words and their counts
    print(t.word_counts)
    #this is a dictionary of words and how many documents they appeared in
    print(t.document_count)
    #this is a dictioanry of words and their uniquely assigned integers
    print(t.word_index)
    #this is an integer count of the total number of documents that were used to fit the tokenizer
    print(t.word_docs)

    #integer encode the documents (one hot encoding in this case i believe)
    #one hot encoding does not car about word order, it loses the context we often need
    #but is still a very successful approach
    encoded_docs = t.texts_to_matrix(text, mode='count')
    print(encoded_docs)

    #once we have a tokenizer that we have fit on training data we can use it to encode documents in train or test datasets


    return t, encoded_docs

#scikitBag("data/damonText.txt")
kerasBag("data/damonText.txt")
