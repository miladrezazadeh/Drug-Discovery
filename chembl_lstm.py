## Name: Milad Rezazadeh
## SciNet username: rezaza10
## Description:
## Build a Recurrent Neural Network(RNN), using LSTMs which would take SMILES-format strings as inputs
## and predict each chemical's value of the "AlogP" column -
## a measure of molecular hydrophobicity (lipophilicity).

#######################################################################################

## Import required libraries
import numpy as np
import pandas as pd
import numpy as np
import keras.models as km
import keras.layers as kl
import sklearn.model_selection as skms
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings('always')

print("Using Tensorflow backend.")

#######################################################################################

## Reading the input files

df = pd.read_csv('chembl.csv')
print("Reading ChEMB data.")

## Using portion of all data
x_data = df['Smiles'][:100000,]
y_data = df['AlogP'][:100000,]

#######################################################################################


"""
Data preprocessing:

Build a character set from all characters found in the SMILES string.
Start and stop characters are added (E and !).The stop character also
work as padding to get the same length of all vectors, so that the network
can be trained in batch mode.

Then, Transform each SMILES sequence into a sequence of one-hot encoded vectors.

this snippet of code is derived from original source code:
https://www.cheminformania.com/master-your-molecule-generator-seq2seq-rnn-models-with-smiles-in-keras/


"""

charset = set("".join(list(df.Smiles))+"!E")
## Convert char to int for encoding the SMILEs
char_to_int = dict((c,i) for i,c in enumerate(charset))
## Decoding the int to char(good if we need decoder)
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in df.Smiles]) + 5
# print(str(charset))
# print(len(charset), embed)

## Function to one-hot encode SMILEs string
def vectorize(smiles):
    one_hot = np.zeros((smiles.shape[0], embed, len(charset)), dtype=np.int8)
    for i, smile in enumerate(smiles):
        # encode the startchar
        one_hot[i, 0, char_to_int["!"]] = 1
        # encode the rest of the chars
        for j, c in enumerate(smile):
            one_hot[i, j + 1, char_to_int[c]] = 1
        # Encode endchar
        one_hot[i, len(smile) + 1:, char_to_int["E"]] = 1
    # Return two, one for input and the other for output
    return one_hot[:, 0:-1, :]

## Assign encoded data to another variable
X_enc = vectorize(x_data.values)

#######################################################################################

## split data into training and test
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X_enc, y_data, test_size=0.25, random_state=42)

#######################################################################################

## Create RNN-LSTM
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

input_dim = train_X.shape[1:]
output_dim = train_y.shape[-1]

print('Building network.')

model = km.Sequential()
## A layer of LSTMs
model.add(kl.LSTM(256, input_shape = input_dim))
## Add fully-connected output layer
model.add(kl.Dense(output_dim, activation = 'linear'))


## compilation.
model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])

## Fiting the model
print("Training network.")
history = model.fit(train_X, train_y, batch_size=128, epochs=100)

## score of the training dataset
print("The training score is ", model.evaluate(train_X, train_y))

## score of the testing dataset
print("The test score is ", model.evaluate(test_X, test_y))
