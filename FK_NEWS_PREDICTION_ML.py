#!/usr/bin/env python
# coding: utf-8

# # FAKE NEWS DETECTION (using regression)

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[2]:


df_fake = pd.read_csv('C:/Users/user/Favorites/My_FN_project/Fake.csv')
df_true = pd.read_csv('C:/Users/user/Favorites/My_FN_project/True.csv')


# In[3]:


df_fake.head(5)   #to show first 5 results


# In[4]:


df_true.head(5)    #to show first 5 results


# In[5]:


df_fake["class"] = 0    #Inserting a column called "class" for fake and real news dataset to categories fake and true news.
df_true["class"] = 1


# In[6]:


df_fake.shape , df_true.shape


# In[7]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[8]:


df_fake.shape, df_true.shape


# In[9]:


df_fake_manual_testing["class"] = 0  #Merging the manual testing dataframe in single dataset and save it in a csv file
df_true_manual_testing["class"] = 1


# In[10]:


df_fake_manual_testing.head(10) #new fake manual testing data


# In[11]:


df_true_manual_testing.head(10)  #new true manual testing data


# In[12]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("C:/Users/user/Favorites/My_FN_project/manual_testing.csv")


# In[13]:


df_marge = pd.concat([df_fake, df_true], axis =0 )  #Merging the main fake and true dataframe
df_marge.head(10)


# In[14]:


df_marge.columns


# In[15]:


df = df_marge.drop(["title", "subject","date"], axis = 1) #"title", "subject" and "date" columns is not required for detecting the fake news, so dropping the columns.


# In[16]:


df.isnull().sum()  #eliminate null


# In[17]:


df = df.sample(frac = 1)  #randomly shuffling the data


# In[18]:


df.head() #desired dataset


# In[19]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


# In[20]:


df.columns


# In[21]:


df.head()


# In[22]:


def wordopt(text):  #Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links.
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[23]:


df["text"] = df["text"].apply(wordopt)


# In[24]:


x = df["text"]  #Defining dependent and independent variable as x and y
y = df["class"]


# In[25]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25) #Splitting the dataset into training set and testing set.


# In[26]:


from sklearn.feature_extraction.text import TfidfVectorizer  #convert text into vectors


# In[27]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[28]:


#1. Logistic Regression
from sklearn.linear_model import LogisticRegression


# In[29]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[30]:


pred_lr=LR.predict(xv_test)


# In[31]:


LR.score(xv_test, y_test)


# In[32]:


print(classification_report(y_test, pred_lr))


# In[33]:


#2. Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier


# In[34]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[35]:


pred_dt = DT.predict(xv_test)


# In[36]:


DT.score(xv_test, y_test)


# In[37]:


print(classification_report(y_test, pred_dt))


# In[38]:


#3. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier


# In[39]:


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)


# In[40]:


pred_gbc = GBC.predict(xv_test)


# In[41]:


GBC.score(xv_test, y_test)


# In[42]:


print(classification_report(y_test, pred_gbc))


# In[43]:


#4. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


# In[44]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# In[45]:


pred_rfc = RFC.predict(xv_test)


# In[46]:


RFC.score(xv_test, y_test)


# In[47]:


print(classification_report(y_test, pred_rfc))


# ## MODEL TESTING (MANUAL ENTRY)

# In[48]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# In[49]:


news = str(input())
manual_testing(news)


# In[ ]:




