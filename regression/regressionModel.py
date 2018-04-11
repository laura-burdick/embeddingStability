# Author: Laura Wendlandt
# Dependencies: sklearn, tqdm
# Train regression model

import pickle
from sklearn.linear_model import Ridge
from tqdm import tqdm

# @TODO Change these variables before running
dataFolder = '/local/embedding_datasets/metaClassifier/' #Location of output feature files
pickledFiles = ['all_features_take2_0_500','all_features_take2_500_1000','all_features_take2_1000_1500','all_features_take2_1500_2000','all_features_take2_2000_2500','all_features_take2_2500_2521'] #all output feature files
OUTPUT_FILE = dataFolder + 'regressionModel_take2.pkl' #Save regression model here
TRAINING_DATA_FILE = dataFolder + 'X.pkl' #Where to save the training data (used for calculating R^2 later on)
TRAINING_TARGET_FILE = dataFolder + 'y.pkl' #Where to save the labels for the training data (used for calculating R^2 later on)

print('Reading in all features...')
all_features = {}
for name in tqdm(pickledFiles):
    with open(dataFolder+name+'.pkl','rb') as pickleFile:
        all_features.update(pickle.load(pickleFile))

print('Creating model...')
X = []
y = []
for word,properties in tqdm(all_features.items()):
    features = properties[0]
    stability = properties[1]
    X.append(features)
    y.append(stability)
with open(TRAINING_DATA_FILE,'wb') as pickleFile:
	pickle.dump(X,pickleFile)
with open(TRAINING_TARGET_FILE,'wb') as pickleFile:
	pickle.dump(y,pickleFile)

model = Ridge(random_state=42)

print('Fitting model...')
model.fit(X,y)

print('Saving model...')
with open(OUTPUT_FILE,'wb') as pickleFile:
    pickle.dump(model,pickleFile)
