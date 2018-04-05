# Author: Laura Wendlandt (wenlaura@umich.edu)
# Dependencies: scipy
# Train 5 embedding spaces (5 embedding sizes) using PPMI on Europarl

import numpy as np
import pickle
import math
from scipy import sparse
from scipy.sparse.linalg import svds

# @TODO Change these variables before running
WORDLIST_FILE = '/local/embedding_datasets/europarl/wordList.pkl' #Location of pickled wordlist (see getWordList.py)
EUROPARL_FILE = '/local/embedding_datasets/europarl/europarl_processed.txt' #All of English Europarl, one sentence per line, already tokenized with tokens separated by spaces
OUTPUT_FOLDER = '/local/embedding_datasets/europarl/metaClassifier/' #Where the embedding spaces should be saved

def buildCooccurrenceMatrix(sentences,wordList,windowSize=5):
   print('Building cooccurence matrix...')
   cooccur = np.zeros((len(wordList)+1,len(wordList)+1)) #Last dimension is for unknown words
   for sentence in sentences:
      sentence = [wordList.index(i) if i in wordList else len(wordList) for i in sentence]
      for i in range(len(sentence)-1):
         for j in range(1,windowSize):
            if i+j >= len(sentence):
               break
            cooccur[sentence[i]][sentence[i+j]] += 1
            if sentence[i+j] != sentence[i]:
               cooccur[sentence[i+j]][sentence[i]] += 1
   
   return cooccur

def getPPMI(cooccur,wordList):
   dMagnitude = 0
   singleCounts = []
   for i in range(len(cooccur)):
      count = 0
      for j in range(len(cooccur)):
         count += cooccur[i][j]
      singleCounts.append(count)
      dMagnitude += count
      
   #Calculate ppmi for all entries in cooccur
   ppmi = np.zeros((len(cooccur),len(cooccur)))
   for i in range(len(cooccur)):
      for j in range(len(cooccur)):
         if i < j: #will be a symmetric matrix 
            break
         if cooccur[i][j] > 0:
            pmi = math.log((cooccur[i][j]*dMagnitude)/(singleCounts[i]*singleCounts[j]),2)
            if pmi > 0:
               ppmi[i][j] = ppmi[j][i] = pmi

   print('Converting to sparse matrix...')
   ppmi = sparse.csr_matrix(ppmi)

   print('Calculating SVD...')
   #Return full SVD
   #NOTE: If your vocabulary size is less than 800, the next line won't work. Change k to be less than 800 dimensions (and less than your vocabulary size).
   #Here, k is set to be the largest embedding dimension that we want to consider.
   (U,S,V) = svds(ppmi,k=800)

   return (U,S)

def train_ppmi(U,S,wordList,embeddingSize,name):
    filenames = []
    
    #Train model
    model = U[:,:embeddingSize].dot(np.diag(S[:embeddingSize]))
    
    #Save model for future use
    filename = name + 'ppmi_' + str(embeddingSize)
    filenames.append(filename)
    save_dict = {}
    for i in range(len(wordList)):
        save_dict[wordList[i]] = model[i]
    with open(OUTPUT_FOLDER+filename+'.pkl', 'wb') as pickle_file:
        pickle.dump(save_dict,pickle_file)
        
    return filenames

print('Loading data...')
with open(EUROPARL_FILE,'r') as europarl:
   sentences = europarl.readlines()
   sentences = [i[:-1].split(' ') for i in sentences]

print('getWordList...')
with open(WORDLIST_FILE,'rb') as pickleFile:
   wordList = pickle.load(pickleFile)

print('buildCooccurrenceMatrix...')
cooccur = buildCooccurrenceMatrix(sentences,wordList)
print('getPPMI...')
(U,S) = getPPMI(cooccur,wordList)

print('Training PPMI...')
embedding_sizes = [50,100,200,400,800]
for size in embedding_sizes:
    print(size)
    train_ppmi(U,S,wordList,size,'')

