# Author: Laura Wendlandt (wenlaura@umich.edu)
# Creates a wordlist for a given text file

from collections import Counter
import pickle

# @TODO Change these variables before running
TEXT_FILE = '/local/embedding_datasets/europarl/europarl_processed.txt' #Text file, one sentence per line, already tokenized with tokens separated by spaces
OUTPUT_FILE = '/local/embedding_datasets/europarl/europarl_wordList.pkl' #Where the pickled wordlist should be saved

# Generate word list from training data
#
# @param sentences
#    A list of sentences, tokenized on word (already preprocessed)
# @param minCount
#    Ignore words with a count lower than this
#
# @returns a list of all of the words in the corpus
#
def getWordList(sentences,minCount=5):
    wordList = []
    count = Counter()
    for sentence in sentences:
        for word in sentence:
            count[word] += 1
    for word,c in count.items():
        if c >= minCount:
            wordList.append(word)
    return wordList

with open(TEXT_FILE,'r') as europarl:
	sentences = europarl.readlines()
	sentences = [i[:-1].split(' ') for i in sentences]

print('Getting word list...')
wordList = getWordList(sentences)

print('Saving word list...')
with open(OUTPUT_FILE,'wb') as pickleFile:
	pickle.dump(wordList,pickleFile)
