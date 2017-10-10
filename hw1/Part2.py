
# coding: utf-8

# ## Importing Libraries

# In[1]:

#Importing Libraries
import pandas as pd
import numpy as np
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras.python.keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from collections import Counter




# ## Loading Data

# In[2]:

#Loading files
body=[]
path = 'D:/Share Folder/Pace 3rd Sem/Deep Learning/Assignments/*.txt'   
files=glob.glob(path)

for file in files:
    f=open(file, 'r')
    content=f.read()
    body.append(content)
    f.close()

body_file=pd.DataFrame(data=body,columns=['bodies'])
labels_file=pd.read_csv("D:\Share Folder\Pace 3rd Sem\Deep Learning\labels.txt",header=None,delimiter=" ",names=['label','fname'])

#Merging dataframes
final_df=pd.concat([body_file,labels_file],axis=1)
final_df.head()


# ## Splitting the Data 

# In[3]:

np.random.seed(123)
msk = np.random.rand(len(final_df)) < 0.7
train = final_df[msk]
test = final_df[~msk]
len(train),len(test)


# In[4]:

email_x_train=train.bodies
email_y_train=train.label
email_x_test=test.bodies
email_y_test=test.label


# ## BOW formation

# In[5]:

max_words = 100
print (max_words)
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
#print(tokenize)


# In[6]:

tokenize.fit_on_texts(email_x_train) # only fit on train
x_train = tokenize.texts_to_matrix(email_x_train)
x_test = tokenize.texts_to_matrix(email_x_test)


# In[7]:

encoder=LabelEncoder()
encoder.fit(email_y_train)
y_train=encoder.transform(email_y_train)
y_test=encoder.transform(email_y_test)


# In[8]:

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# ## Implementation of KNN

# In[9]:

def train(x_train, y_train):
	return


def predict(x_train, y_train, x_test, k):
    distances = []
    targets = []
    
    for i in range(len(x_train)):
        distance = np.sqrt(np.sum(np.square(x_test - x_train[i, :])))
        distances.append([distance, i])
        
    
    distances = sorted(distances)
    #print(distances)

    for i in range(k):
        index = distances[i][1]
        #print(index)
        targets.append(y_train[index])
    #print(targets)
    return Counter(targets).most_common(1)[0][0]


def kNearestNeighbor(x_train, y_train, x_test, y_pred, k):
    # train on the input data
    train(x_train, y_train)
    #print(type(x_test))

    # loop over all observations
    for i in range(len(x_test)):
        y_pred.append(predict(x_train, y_train, x_test[i, :], k))

y_pred = []


# ## Applying KNN model

# In[10]:

kNearestNeighbor(x_train, y_train, x_test, y_pred, 3)
y_pred = np.asarray(y_pred)
print("Predicted Result: %s " %(y_pred))


# ## Accuracy Score

# In[11]:

knnacc = accuracy_score(y_test, y_pred)
print ("The accuracy of model is %s" %(knnacc))


# ## Confusion Matrix

# In[12]:

cnf = confusion_matrix(y_test, y_pred)
print ((cnf))

