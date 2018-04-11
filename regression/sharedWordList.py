# Author: Laura Wendlandt
# Get shared word list (all words that occur in all of the domains)

import pickle

# @TODO Change theses variables before running
datasetFolder='/local/embedding_datasets/' #Base path
NYT_FOLDER = datasetFolder+'nyt_corpus/processed/' #All NYT domain data is here, one sentence per line, already tokenized with tokens separated by spaces
EUROPARL_FILE = datasetFolder+'europarl/europarl_processed.txt' #All of English Europarl, one sentence per line, already tokenized with tokens separated by spaces
NYT_WORDLIST_FOLDER = datasetFolder+'nyt_corpus/metaClassifier/' #Location of all NYT wordlists (see getWordList.py)
EUROPARL_WORDLIST_FILE = datasetFolder+'europarl/wordList.pkl' #Location of Europarl wordlist
OUTPUT_FILE = datasetFolder+'nyt_corpus/metaClassifier/combinedWordList.pkl' #Output shared word list

print('Prepare all training data...')
#Prepare all training data
domains = ['U.S.','New_York_and_Region','Business','Arts','Sports']
data = {}
wordlists = []
for domain in domains:
    print(domain)
    with open(NYT_FOLDER+domain+'.data','r') as domainFile:
        sentences = domainFile.readlines()
        data[domain] = [i[:-1].split(' ') for i in sentences] #tokenize on space
    with open(NYT_WORDLIST_FOLDER+domain+'_wordList.pkl','rb') as pickleFile:
        wordlists.append(pickle.load(pickleFile))

print('Europarl')
domains.append('Europarl')
with open(EUROPARL_FILE,'r') as europarl:
    sentences = europarl.readlines()
    data['Europarl'] = [i[:-1].split(' ') for i in sentences]
with open(EUROPARL_WORDLIST_FILE,'rb') as pickleFile:
    wordlists.append(pickle.load(pickleFile))

combined = set(wordlists[0]) & set(wordlists[1]) & set(wordlists[2]) & set(wordlists[3]) & set(wordlists[4]) & set(wordlists[5])
print(len(combined))
with open(OUTPUT_FILE,'wb') as pickleFile:
    pickle.dump(list(combined),pickleFile)
