import pandas as pd 
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_initial_data(train_data_path, test_data_path):
    df=pd.read_csv(train_data_path)
    test=pd.read_csv(test_data_path)
    text=df['tweet'].values.tolist()
    text_test=test['tweet'].values.tolist()
    text+=text_test

    stopword=nltk.corpus.stopwords.words('english')
    stopword.remove('not')
    for index,text_ in enumerate(text):
        text_=re.sub(r'@[\w]*','',text_) #Removing Twitter Handles (@user)
        text_=re.sub(r'http/S+','',text_) #Removing urls from text 
        text_=re.sub(r'[^A-Za-z#]',' ',text_) #Removing Punctuations, Numbers, and Special Characters
        text_=" ".join(i.lower() for i in text_.split() if i.lower() not in stopword) #Removing stopword
        text[index]=text_

    #Stemming the word
    pt=PorterStemmer()
    wordnet=WordNetLemmatizer()
    for index,text_ in enumerate(text):
        text_=" ".join(pt.stem(i) for i in text_.split())
        text_=" ".join(wordnet.lemmatize(i) for i in text_.split())  
        text[index]=text_

    df['preprocess_tweet']=text[:len(df)]
    df['length_tweet']=df['preprocess_tweet'].str.len()
    test['preprocess_tweet']=text[len(df):]

    train=df.copy()
    train.drop(columns=['id','tweet','preprocess_tweet'],inplace=True)

    bow=CountVectorizer( min_df=2, max_features=1000)
    bow.fit(df['preprocess_tweet'])
    bow_df=bow.transform(df['preprocess_tweet']).toarray()
    print('feature name==',bow.get_feature_names()[:10])
    print('number of uniqe words',bow_df.shape[1])
    print('shape',bow_df.shape)
    bow_train=pd.DataFrame(bow_df)
    bow_train['length_tweet']=df['length_tweet']
    bow_train['label']=df['label']

    major_class_0,major_class_1=bow_train.label.value_counts()
    df_major=bow_train[bow_train['label']==0]
    df_minor=bow_train[bow_train['label']==1]
    df_minor_upsampled = resample(df_minor, 
                                    replace=True,     # sample with replacement
                                    n_samples=major_class_0)
    df_bow_upsampled = pd.concat([df_major, df_minor_upsampled])

    x=df_bow_upsampled.iloc[:,0:-1]
    y=df_bow_upsampled['label']
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

    return X_train, X_test, y_train, y_test