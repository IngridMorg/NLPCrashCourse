#this is from lesson 7 of:
#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

#in this project i will use an 80% train and 20% test set approach
#with a 50/50 split in the test set between positive and negative reviews

#need to clean and tokenise the sets (we will save this to a file perhaps)

#i need to clean and tokenise all the data
import string

sentences = []
text = []

def loadText(file):
    # open file and read text into the variable 'text
    file = open(file, encoding="utf-8")

    text = file.read()
    file.close()
    # split the words based on newline characters
    text = text.splitlines()
    for i in range(0,len(text)):
        text[i] = text[i].translate(str.maketrans('', '', string.punctuation))
        text[i] = text[i].lower()
    return text

text = loadText("data/Neg.txt")
print(text)
