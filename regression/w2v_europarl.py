# Author: Laura Wendlandt (wenlaura@umich.edu)
# Dependencies: gensim
# Train 25 embedding spaces (5 embedding sizes, 5 random seeds each) using word2vec on Europarl

from collections import Counter
from gensim.models import word2vec
import sys
import pickle

# @TODO Change these variables before running
WORDLIST_FILE = '/local/embedding_datasets/europarl/wordList.pkl' #Location of pickled wordlist (see getWordList.py)
EUROPARL_FILE = '/local/embedding_datasets/europarl/europarl_processed.txt' #All of English europarl, one sentence per line, already tokenized with tokens separated by spaces
OUTPUT_FOLDER = '/local/embedding_datasets/europarl/metaClassifier/' #Where the embedding spaces should be saved

with open(EUROPARL_FILE,'r') as europarl:
	sentences = europarl.readlines()
	sentences = [i[:-1].split(' ') for i in sentences]

with open(WORDLIST_FILE,'rb') as pickleFile:
	wordList = pickle.load(pickleFile)

sizes = [50,100,200,400,800]
seeds = [2518,2548,2590,29,401]
for size in sizes:
	print('size',size)
	for seed in seeds:
		print(seed)
		model = word2vec.Word2Vec(sentences,size=size,window=5,min_count=5,seed=seed)

		#Save model for future use
		filename = 'w2v_' + str(size) + '_' + str(seed)
		save_dict = {}
		for word in wordList:
			if word in model:
				save_dict[word] = model[word]
		with open(OUTPUT_FOLDER+filename+'.pkl', 'wb') as pickle_file:
			pickle.dump(save_dict,pickle_file)
