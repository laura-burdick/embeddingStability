# Author: Laura Wendlandt
# Dependencies: sklearn, numpy, tqdm
# Pre-calculate ten nearest neighbors for all words in all Europarl embedding spaces

import pickle
from sklearn.neighbors import BallTree
import numpy as np
from tqdm import tqdm,trange

# @TODO Change these variables before running
WORDLIST_FILE = '/local/embedding_datasets/europarl/metaClassifier/wordList.pkl' #Location of pickled wordlist (see getWordList.py)
DISTANCES_FOLDER = '/local/embedding_datasets/europarl/metaClassifier/' #Location of distance matrices
BALLTREE_FOLDER = '/local/embedding_datasets/europarl/metaClassifier/balltree_leaf2_fast/' #Location of balltrees
OUTPUT_FOLDER = '/local/embedding_datasets/europarl/metaClassifier'

def mydist(x,y):
	return distances[int(x[0])][int(y[0])]

with open(WORDLIST_FILE,'rb') as pickleFile:
	wordList = pickle.load(pickleFile)

for algorithm in ['w2v','glove','ppmi']:
	for dimension in [50,100,200,400,800]:
		for seed in [2518,2548,2590,29,401]:
			name = algorithm + '_' + str(dimension) + '_' + str(seed)
			if algorithm == 'ppmi':
				name = algorithm + '_' + str(dimension)
			print(name)
		
			print('Loading distances...')
			distances = np.zeros((len(wordList),len(wordList)))
			for index in trange(1000,len(wordList)+999,1000):
				filename = DISTANCES_FOLDER+'distances_' + name + '_i' + str(index) + '.pkl'
				with open(filename,'rb') as pickleFile:
					if index > len(wordList):
						distances[index-1000:len(wordList),:] = pickle.load(pickleFile)
					else:
						distances[index-1000:index,:] = pickle.load(pickleFile)
			
			print('Loading balltree...')
			with open(BALLTREE_FOLDER+name+'_balltree.pkl','rb') as pickleFile:
				balltree = pickle.load(pickleFile)

			print('Calculating nearest neighbors...')
			nearestNeighbors = {}
			for i in tqdm(range(len(wordList))):
				dist,ind = balltree.query(np.array([i]).reshape(1,-1),k=11)
				nearestNeighbors[wordList[i]] = [wordList[k] for k in ind[0][1:]]

			print('Saving nearest neighbors...')
			with open(OUTPUT_FOLDER+'tenNearestNeighbors_'+name+'.pkl','wb') as pickleFile:
				pickle.dump(nearestNeighbors,pickleFile)
