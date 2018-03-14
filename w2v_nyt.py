# Author: Laura Wendlandt (wenlaura@umich.edu)
# Dependencies: gensim
# Train 25 embedding spaces (5 embedding sizes, 5 random seeds each) using word2vec on NYTcorpus
# Note: to reproduce the regression model, this script must be run 6 different times (one for each different NYT domain)

from collections import Counter
from gensim.models import word2vec
import sys
import pickle
import string

# @TODO Change these variables before running
WORDLIST_FILE = '/local/embedding_datasets/nyt_corpus/wordList.pkl' #Location of pickled wordlist (see getWordList.py)
NYT_DOMAIN = 'all' #Which NYT domain to train embedding spaces for
NYT_FILE = '/local/embedding_datasets/nyt_corpus/processed/top5.data' #NYT domain data, one sentence per line, already tokenized with tokens separated by spaces
OUTPUT_FOLDER = '/local/embedding_datasets/nyt_corpus/metaClassifier/' #Where the embedding spaces should be saved

with open(NYT_FILE,'r') as nyt:
	sentences = nyt.readlines()
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
		filename = NYT_DOMAIN+'_w2v_' + str(size) + '_' + str(seed)
		save_dict = {}
		for word in wordList:
			if word in model:
				save_dict[word] = model[word]
		with open(OUTPUT_FOLDER+filename+'.pkl', 'wb') as pickle_file:
			pickle.dump(save_dict,pickle_file)
