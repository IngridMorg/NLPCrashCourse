#this is for loading data from files, cleaning and tokenising the data and then
#saving/loading to a file
import os
import string

def loadAll(directory, type):
     for f in os.listdir(directory):
         t = loadData(f,type)
         saveData(t,type)

def loadData(filename, type):
     if(type == 2):
          file = open("data/neg/"+filename, encoding='utf-8')
     if(type == 1):
          file = open("data/pos/"+filename, encoding='utf-8')
     text = file.read()
     file.close()
     # split the words based on newline characters
     text = text.splitlines()
     for i in range(0, len(text)):
          text[i] = text[i].translate(str.maketrans('', '', string.punctuation))
          text[i] = text[i].lower()
     return text

def saveData(text, type):
     #save to the positive file
     if(type == 1):
          file = open("data/Pos.txt","a")
          for t in text:
               file.write(t)
               file.write("\n")

     #save to the negative file
     elif(type == 2):
          file = open("data/Neg.txt","a")
          for t in text:
               file.write(t)
               file.write("\n")

     file.close()

loadAll("data/neg",2)