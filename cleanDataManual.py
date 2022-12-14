#lesson 2
#https://machinelearningmastery.com/crash-course-deep-learning-natural-language-processing/

#this is the manual approach to word tokenisation
import string

def Clean(filename):
    #open file and read text into the variable 'text
    file = open(filename,encoding="utf-8")

    text = file.read()
    file.close()

    #split the words based on white space
    words = text.split()

    #make everything lower case
    words = [w.lower() for w in words]
    #remove punctuation
    translator = str.maketrans('','',string.punctuation)
    for i in range(0,len(words)):
        words[i] = words[i].translate(str.maketrans('','',string.punctuation))
        #we have to use this section as the downloaded text uses symbols not contained within string.punctuation
        words[i] = words[i].replace('“','')
        words[i] = words[i].replace('’', '')
        words[i] = words[i].replace('”','')
        words[i] = words[i].replace('\n', '')
    #i think im going to leave in numbers, they can be useful
    return words