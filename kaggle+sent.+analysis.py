
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


os.getcwd()


# In[3]:


os.chdir('C:\\Users\\Sharath\\Desktop')


# In[4]:


df=pd.read_csv('labeledTrainData.tsv',header=0,delimiter='\t',quoting =3)


# In[5]:


df.shape


# In[6]:


df.columns.values


# In[7]:


df.head()


# In[8]:


df['review'][0]


# In[9]:


from bs4 import BeautifulSoup as bs


# In[10]:


exmp=bs(df['review'][0])


# In[11]:


import re


# In[12]:


letters_only=re.sub("[^a-zA-Z]"," ",exmp.get_text())
print (letters_only)


# In[13]:


lower_case=letters_only.lower()
words=lower_case.split()


# In[14]:


from nltk.corpus import stopwords


# In[15]:


print(stopwords.words('english'))


# In[16]:


sw=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'nor','only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']


# In[17]:


filtered=[w for w in words if not w in sw]


# In[18]:


len(words)


# In[19]:


len(filtered)


# In[20]:


def review_to_words(x):
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", x) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in sw]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


# In[21]:


df['review']=df['review'].apply(review_to_words)


# In[22]:


num_reviews=df['review'].size


# In[23]:


num_reviews


# In[24]:


clean_train_reviews = []
for i in range( 0, num_reviews ):
    if( (i+1)%1000 == 0 ):
        print ("Review %d of %d\n" % ( i+1, num_reviews ))
    clean_train_reviews.append( review_to_words( df["review"][i] ) )


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer


# In[26]:


vectorizer = CountVectorizer(analyzer = "word",                                tokenizer = None,                                 preprocessor = None,                              stop_words = None,                                max_features = 5000)


# In[27]:


train_data_features = vectorizer.fit_transform(clean_train_reviews)


# In[28]:


train_data_features = train_data_features.toarray()


# In[29]:


print (train_data_features.shape)


# In[30]:


vocab = vectorizer.get_feature_names()
print (vocab)


# In[31]:


import numpy as np

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it 
# appears in the training set
for tag, count in zip(vocab, dist):
    print (count, tag)


# In[32]:


print ("Training the random forest...")
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100) 

# Fit the forest to the training set, using the bag of words as 
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( train_data_features, df["sentiment"] )


# In[33]:


os.getcwd()


# In[34]:


# Read the test data
test = pd.read_csv("testData.tsv", header=0, delimiter="\t",                    quoting=3 )


# In[35]:


print (test.shape)


# In[36]:


num_reviews = len(test["review"])


# In[37]:


clean_test_reviews = [] 

print ("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,num_reviews):
    if( (i+1) % 1000 == 0 ):
        print ("Review %d of %d\n" % (i+1, num_reviews))
    clean_review = review_to_words( test["review"][i] )
    clean_test_reviews.append( clean_review )


# In[38]:


test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()


# In[39]:


result = forest.predict(test_data_features)


# In[40]:


output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )


# In[44]:


output.head()


# In[46]:


output.to_csv( "Bag_of_Words_modell.csv", index=False, quoting=3 )


# In[43]:


test['review'][2]


# In[47]:


test1=pd.read_csv('req.csv',index_col=0)


# In[49]:


test1.head()


# In[50]:


def permu(x):
    if x['newp'] and x['newn']==1:
        return 0
    elif x['newp']==1:
        return 1
    elif x['newn']==1:
        return 0


# In[51]:


test1['allnew']=test1.apply(permu,axis=1)


# In[53]:


def review_to_words( x ):
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", x) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set               
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in sw]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))


# In[54]:


test1['Body Title']=test1['Body Title'].apply(review_to_words)


# In[55]:


test1['Review Body']=test1['Review Body'].apply(review_to_words)


# In[58]:


num_reviews1 = len(test1["Body Title"])


# In[60]:


num_reviews2 = len(test1["Review Body"])


# In[62]:


clean_test_reviews1 = []
for i in range(0,num_reviews1):
    clean_review = test1["Body Title"][i]
    clean_test_reviews1.append( clean_review )


# In[64]:


clean_test_reviews2 = []
for i in range(0,num_reviews2):
    clean_review = test1["Review Body"][i]
    clean_test_reviews2.append( clean_review )


# In[66]:


test_data_features1 = vectorizer.transform(clean_test_reviews1)
test_data_features2 = vectorizer.transform(clean_test_reviews2)


# In[68]:


test_data_features1 = test_data_features1.toarray()
test_data_features2 = test_data_features2.toarray()


# In[69]:


result1 = forest.predict(test_data_features1)
result2 = forest.predict(test_data_features2)


# In[75]:


output=pd.DataFrame( data={"RT":test1["Body Title"], "sentiment1":result1,"RB":test1["Review Body"], "sentiment2":result2})


# In[76]:


output.to_csv( "sfrbagofwords.csv", index=False)


# In[ ]:




