#lesson 3
#bag-of-words encoding using scikit and python
from sklearn.feature_extraction.text import TfidfVectorizer
#we need to load and tokenise the text
from cleanDataManual import *
def loadText(file):
    # open file and read text into the variable 'text
    file = open(file, encoding="utf-8")

    text = file.read()
    file.close()
    # split the words based on newline characters
    text = text.split("\n")
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
    #print(vectorizer.idf_)
    #now we encode the document
    vector = vectorizer.transform([text[0]])
    #summarise the encoded vector
    #print(vector.shape)
    #print(vector.toarray())
    #print(vector)


scikitBag("data/damonText.txt")

#def kerasBag():
