from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# file_o is the cleaned files, in order of p-value range from smallest to largest
# file_r is the raw data from Voicebase
file_o = open('ssn_data_clean.csv', 'r')
file_r = open('ssn_training_data_11_21_2017.csv', 'r')

file_o = pd.read_csv(file_o)
file_r = pd.read_csv(file_r)

fileIds_o = file_o.fileId
fileIds_r = file_r.fileId


# Manualy creating a word list of interest which would help indicate presence of SSN
words_of_interest = ['social', 'security', 'one', 'two','to', 'too', 'three', 
'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', 'oh', 'digit', 'digits', 
'thanks', 'thank', 'you',  'number', 'numbers', 'please', 'give', 'me', 'your',
 'last', 'first', 'would', 'you', 'twenty', 'thirty', 'fourty', 'fifty', 'sixty', 
'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'full', 
'provide', 'me', 'confirm', 'telephone', 'phone', 
'credit', 'card', 'road', 'street', 'house', 'identification', 'address']

# Putting them into a dictionary for faster iteration
wds_interest = {}
index = 0
for wd in words_of_interest:
    wds_interest[wd] = index
    index += 1


"""
The window_scores function takes as input a txt file and uses a sliding window 
of size 'winsize' which moves by 'moveby' number of words each time. It returns
a score of features for each window based on word overlap with the words of 
interest, and the wndows position. It aslo returns a 0 if there is no SSN 
SSN present and a 1 if there is. 
"""
def window_scores(filename, winsize=15, moveby= 5):

    #filepath = '/home/jsw/Documents/uni/voicebase/txt_files/'
    filepath = 'U:/Uni/Voicebase/txt_files/'
    file = open(filepath + filename, 'r')

    # This code removes the .txt file extensions
    file_id = filename[:-4]
    
    # Creating a list of words as strings
    words = []
    for line in file:
    	line = line.lower()
    	line = line.split(' ')
    	words.append(line)
    
    # Index of the string of words
    words = words[0]
    
    # Setting up the window_score dataframe
    window = np.arange(len(words)/5)
    window_score = pd.DataFrame()
    window_score['Session_id'] = 0
    window_score['WinPos'] = window*moveby
    window_score['SSN'] = 0
    window_score['Session_id'] = file_id
    
    # The sliding window of 'winzise' words
    # New string entered into windows every 'moveby' words (5)
    windows = []
    for i in range(len(words)):
    	if i%moveby == 0 or i == 0:
    		string = ' '.join(words[i:i+(winsize-1)])
    		windows.append(string)

    
    scores_list = []
    for i in range(len(windows)):
        scores = np.zeros(len(words_of_interest)+1)
        # Scoring each word of interest a count of how many times it appears 
        # in a window
        for wd in windows[i].split(' '):
            if wd in wds_interest.keys():
                scores[wds_interest[wd]] += 1
        # Inclusion of the proportional window position in scores
        scores[-1] = i/len(words)
        scores_list.append(scores)

        
   
    window_score['Score'] = scores_list
    
    # Ensuring there are no dupicates or files with p-val ranges more than 30
    file = open('ssn_data_clean.csv', 'r')
    truth = pd.read_csv(file)
    truth = truth.drop_duplicates(subset = 'fileId')
    truth = truth[(truth.label == 'NoSSN') | (truth.endPos - truth.startPos < 30)]
    truth.endPos[truth.label == 'NoSSN'] = 0
    
    # IndexError caused by end position of certain files 
    try:
        start = truth.startPos[truth.fileId == file_id].astype(float)
        end = truth.endPos[truth.fileId == file_id].astype(float)
        start = np.array(start)
        end = np.array(end)
    except IndexError:
        pass
    
    # Checking wether window falls in range of p-value given, and assigning 1
    for i in window_score.WinPos:
        i = int(i)
        if i < start < i + winsize or i < end < i + winsize:
            window_score.SSN[window_score.WinPos == i] = 1


    return window_score


"""
Function that takes as inputs the range of files to look at and the window size.
Returns the accuracy of all the predictions and also the recall of SSNs
"""
def svm_acc(start_file, end_file, winsize=15, moveby=5):
    
    scores = pd.Series()
    SSNs = pd.Series()
    file_sample = fileIds_o[start_file:end_file]
    
    # Getting the window scores and SSN values into format to be used by SVM
    for file in file_sample:
        df1 = window_scores(file +'.txt', winsize, moveby)
        new_score = pd.Series(df1.Score)
        new_SSN = pd.Series(df1.SSN)
        SSNs = SSNs.append(new_SSN,  ignore_index=True)
        scores = scores.append(new_score,  ignore_index=True)


    # Function splits the set into training and testing data with test size 1/3 of total set
    X_train, X_test, y_train, y_test = train_test_split(scores, SSNs, test_size=0.33)
    

    # Converting to list for fitting
    x_train = [i for i in X_train]
    x_test = [i for i in X_test]
        
    
    # Creating classifier and fitting and predicting values
    clf = svm.SVC(kernel = 'rbf')
    clf.fit(x_train, y_train)    
    y_pred = clf.predict(x_test)

    
    acc_whole = accuracy_score(y_test, y_pred)
    print(acc_whole)
    
    # Measuring the accuracy of predictions where there is a SSN. (recall)
    y_true_ssn = y_test[y_test == 1]
    y_pred_ssn = y_pred[y_test == 1]
    
    acc_ssn = accuracy_score(y_true_ssn, y_pred_ssn)
    print(acc_ssn)
    
    return acc_ssn




