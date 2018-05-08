from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import os


file = open('ssn_training_data_11_21_2017.csv', 'r')

file = pd.read_csv(file)
fileIds = file.fileId
file = open('ssn_training_data_11_21_2017.csv', 'r')
truth = pd.read_csv(file)
truth = truth.drop_duplicates(subset = 'fileId')
truth = truth[(truth.label == 'NoSSN') | (truth.endPos - truth.startPos < 30)]
fileIds = truth.fileId
print(fileIds)
# Manualy creating a word list of interest which would help indicate presence of SSN
words_of_interest = ['social', 'security', 'one', 'two','to', 'too', 'three', 'four', 'five', 'six', 
'seven', 'eight', 'nine', 'zero', 'oh', 'digit', 'digits', 
'thanks', 'thank', 'you',  'number', 'numbers', 'please', 'give', 'me', 'your',
 'last', 'first', 'would', 'you', 'twenty', 'thirty', 'fourty', 'fifty', 'sixty', 
'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'full', 
'provide', 'me', 'confirm', 'telephone', 'phone', 
'credit', 'card', 'road', 'street', 'house', 'identification', 'address']

wds_interest = {}
index = 0

for wd in words_of_interest:
    wds_interest[wd] = index
    index += 1


def window_scores(filename):

    filepath = '/home/jsw/Documents/uni/voicebase/txt_files/'
    #filepath = 'U:/Uni/Voicebase/txt_files/'
    #for filename in os.listdir(filepath)[:2]:
    file = open(filepath + filename, 'r')
    #file = open('txt_files/16486_10_02_16_2016_14-06-15_105.txt')
    
    # This code removes the .txt file extensione
    file_id = filename[:-4]
    
    
    # Creating a list of words as strings
    words = []
    for line in file:
    	line = line.lower()
    	line = line.split(' ')
    	words.append(line)
    
    # Index of the string of words
    words = words[0]
    
    
    window = np.arange(len(words)/5)

    
    window_score = pd.DataFrame()
    window_score['Session_id'] = 0
    window_score['WinPos'] = window*5
    window_score['SSN'] = 0
    window_score['Session_id'] = file_id
    
    windows = []
    
    for i in range(len(words)):
    	if i%5 == 0 or i == 0:
    		string = ' '.join(words[i:i+14])
    		windows.append(string)

    scores_list = []
    for i in range(len(windows)):
        scores = np.zeros(len(words_of_interest))
        for wd in windows[i].split(' '):
            if wd in wds_interest.keys():
                scores[wds_interest[wd]] += 1
        scores_list.append(scores)
        
   
    window_score['Score'] = scores_list
    
    
    file = open('ssn_training_data_11_21_2017.csv', 'r')
    truth = pd.read_csv(file)
    truth = truth.drop_duplicates(subset = 'fileId')
    truth = truth[(truth.label == 'NoSSN') | (truth.endPos - truth.startPos < 30)]
    truth.endPos[truth.label == 'NoSSN'] = 0
    
    try:
        start = truth.startPos[truth.fileId == file_id].astype(float)
        end = truth.endPos[truth.fileId == file_id].astype(float)
        start = np.array(start)
        end = np.array(end)
    except IndexError:
        pass

    
    for i in window_score.WinPos:
        i = int(i)
        
        #if i > start > i+15 or i < end < i+15:
        if (start < i  < end) or (start < i +15  < end):
            window_score.SSN[window_score.WinPos == i] = 1

    return window_score


scores = pd.Series()
SSNs = pd.Series()
# only looking at 20 for processig time reasons
count = 0
for file in fileIds[:50]:
    df1 = window_scores(file +'.txt')
    new_score = pd.Series(df1.Score)
    new_SSN = pd.Series(df1.SSN)
    SSNs = SSNs.append(new_SSN,  ignore_index=True)
    scores = scores.append(new_score,  ignore_index=True)
    count+=1
    print('next file', count)


#print(scores[50:100], SSNs[50:100])
# Function splits the set into training and testing data with test size 1/3 of total set
X_train, X_test, y_train, y_test = train_test_split(scores, SSNs, test_size=0.33)

#print(X_train.shape)

x = []
x_test = []
for i in X_train:
    x.append(i)
for i in X_test:
    x_test.append(i) 
    
test_string = ['my', 'social', 'security', 'number', 'is', 'zero', 'four', 'penis', 'seven', 'two', 'social', 'social', 'social', 'social', 'social']

## Reshaping functions so they can be inputting into SVC classifier
#X_train = X_train.astype(float).values.reshape(len(X_train), 1)
#y_train = y_train.values.reshape(len(X_train), 1)
#X_test = X_test.values.reshape(len(X_test), 1)


# Creating classifier and fitting and predicting values
clf = svm.SVC(kernel = 'rbf')
clf.fit(x, y_train)
y_pred = clf.predict(x_test)
y_true = y_test
acc = accuracy_score(y_test, y_pred)
print(acc)

print(y_true[y_true==1])
print(y_pred[y_pred==1])
#print(y_pred)
# Measuring the accuracy of predictions where there is a SSN. 
# Fairly low ~0.03 at the moment, refinement needed to scoring system
y_true_ssn = y_true[y_true == 1]
y_pred_ssn = y_pred[y_true == 1]
acc = accuracy_score(y_true_ssn, y_pred_ssn)
print(acc)

    