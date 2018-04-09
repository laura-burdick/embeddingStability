# Author: Laura Wendlandt
# Dependencies: sklearn, numpy, tqdm
# Pre-calculate ten nearest neighbors for all words in all NYT embedding spaces
# Note: to reproduce the regression model, this script must be run 6 different times (one for each different NYT domain)

import pickle
from sklearn.neighbors import BallTree
import numpy as np
from tqdm import tqdm

# @TODO Change these variables before running
dataFolder = '/local/embedding_datasets/nyt_corpus/metaClassifier/' #Base path
DISTANCES_FOLDER = dataFolder #Location of distance matrices
BALLTREE_FOLDER = dataFolder+'balltree_leaf2_fast/' #Location of balltrees
domain = 'all' #Which NYT domain to pre-calculate nearest neighbors for
WORDLIST_FILE = dataFolder+domain+'_wordList.pkl' #Location of pickled wordlist (see getWordList.py)
OUTPUT_FOLDER = dataFolder

def mydist(x,y):
	return distances[int(x[0])][int(y[0])]

with open(WORDLIST_FILE,'rb') as pickleFile:
	wordList = pickle.load(pickleFile)

for algorithm in ['w2v','glove','ppmi']:
	for dimension in [50,100,200,400,800]:
		for seed in [2518,2548,2590,29,401]:
			name = domain + '_' + algorithm + '_' + str(dimension) + '_' + str(seed)
			if algorithm == 'ppmi':
				name = domain + '_' + algorithm + '_' + str(dimension)
			print(name)
		
			print('Loading distances...')
			if domain != 'all':
				with open(DISTANCES_FOLDER+'distances_'+name+'.pkl','rb') as pickleFile:
					distances = pickle.load(pickleFile)
			else: #domain == 'all'
				distances = np.zeros((len(wordList),len(wordList)))
				for index in tqdm(range(1000,len(wordList)+999,1000)):
					filename = DISTANCES_FOLDER+'distances_'+name+'_i'+str(index)+'.pkl'
					with open(filename,'rb') as pickleFile:
						if index > len(wordList):
							distances[index-1000:len(wordList),:] = pickle.load(pickleFile)
						else:
							distances[index-1000:index,:] = pickle.load(pickleFile)

			print('Loading balltree...')
			with open(BALLTREE_FOLDER+name+'_balltree.pkl','rb') as pickleFile:
				balltree = pickle.load(pickleFile)

			print('Calculating nearest neighbors...')
			# NOTE: This doesn't work with Python 2.7 - see https://github.com/scikit-learn/scikit-learn/issues/4360
			nearestNeighbors = {}
			for i in tqdm(range(len(wordList))):
				dist,ind = balltree.query(np.array([i]).reshape(1,-1),k=11)
				nearestNeighbors[wordList[i]] = [wordList[k] for k in ind[0][1:]]

			print('Saving nearest neighbors...')
			with open(OUTPUT_FOLDER+'tenNearestNeighbors_'+name+'.pkl','wb') as pickleFile:
				pickle.dump(nearestNeighbors,pickleFile)
