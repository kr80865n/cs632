
# coding: utf-8

# In[8]:



# In[29]:


import sys
import numpy as np
import random
import os
import keras
from keras.models import load_model

#constants
CAT_OUTPUT_LABEL = 1
DOG_OUTPUT_LABEL = 0


TEST_FILE = "validation.npy"


OUT_FILE = "predictions.txt"
BATCH_SIZE = 20

data = np.load(TEST_FILE).item()

train_X = images = data["images"]

if "ids" in data:
    ids = data["ids"]
else:
    ids = list(range(0,len(images)))

model_load = load_model("my_model.h5")

pred = model_load.predict(train_X, BATCH_SIZE, verbose=1)
print(predictions)

out = open(OUT_FILE, "w")
#counter=0;
for i, image in enumerate(images):
    image_id = ids[i]
    prediction = round(float(pred[i]))
    '''if data['labels'][i]==prediction:
        counter+=1'''
    line = str(image_id) + " " + str(prediction) + "\n"
    out.write(line)
out.close()
#print(counter)


# In[ ]:



