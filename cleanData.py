#lesson 2
#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

from nltk.tokenize import word_tokenize
#nltk approach
import string
#open file and read text into the variable 'text
filename = 'data/metamorphosis.txt'
file = open(filename,encoding="utf-8")

text = file.read()
file.close()
#remove punctuation first

#tokenise the words
tokens = word_tokenize(text)
for i in range(0,len(tokens)):
    tokens[i] = tokens[i].translate(str.maketrans('','',string.punctuation))
    #we have to use this section as the downloaded text uses symbols not contained within string.punctuation
    tokens[i] = tokens[i].replace('“','')
    tokens[i] = tokens[i].replace('’', '')
    tokens[i] = tokens[i].replace('”','')

#remove the empty string
while("" in tokens):
    tokens.remove("")

for i in range(0,len(tokens)):
    print(tokens[i])
print(tokens)




