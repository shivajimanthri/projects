
import re
import pickle
import nltk
import streamlit as st
# from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from nltk.tokenize import word_tokenize
from keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer


stopwords = {'mustn', 'which', 'here', 've', 'too', "should've", 'out', 'into', 'if', 'mightn', 'themselves', 'other', 'so', 'won', 'why', 'does', 'same', 'this', 'only', 'all', 'was', 'don', 'over', 'now', 'my', 'between', "she's", 'further', 'being', 'what', 'y', 'doesn', 'yours', 'yourself', 'our', 'you', 'couldn', 'shan', "that'll", 'himself', 'weren', 'has', 'yourselves', 'aren', 'ourselves', 'isn', 'there', 'on', "you'd", 'because', 'once', 't', 'for', 'until', 'itself', 'than', 'i', 'where', 'own', "it's", 'be', 'she', 'or', 'and', 'were', 'hasn', 'these', 'that',  'wasn', 'him', 'had', 'o', 'd', 'through', 'more', 'have', 'will', 'been', "you're", 'hadn', 'do', 'myself', 'when', 'a', 'the', 'he', 'very', 'under', 'ma', 'nor', 'of', 're', 'an', 's', 'are', 'it', 'am', 'again', 'against', 'any', 'we', 'them', 'up', 'ours', 'how', 'your', 'above', 'll', 'shouldn', 'as', 'should', 'to', 'herself', "you'll", "you've", 'did', 'didn', 'who', 'with', 'wouldn', 'no', 'is', 'their', 'whom', 'while', 'both', 'each', 'needn', 'about', 'during', 'down', 'her', 'm', 'at', 'in', 'before', 'they', 'his', 'haven', 'having', 'just', 'doing', 'from', 'those', 'such', 'me', 'after', 'by', 'hers', 'then', 'ain', 'its', 'can', 'theirs', 'off'}

model = load_model('bilstm_model.h5')

max_length = 200
tokenizer_path = 'tokenizer.pkl'
with open (tokenizer_path,'rb') as handle:
    tokenizer = pickle.load(handle)

stemmer = PorterStemmer()

def Get_sentiment(text):
    # text = text.lower()
    ##
    cleaned_text = ' '.join([word for word in text.split() if word.lower() not in stopwords])
    ##
    clean_html_text = re.sub('<[^<]+?>', '', cleaned_text)
    
    tokens = word_tokenize(clean_html_text)
    cleaned_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if re.sub(r'[^\w\s]', '', token)]
    cleaned_text_punc = ' '.join(cleaned_tokens)
    
    word_tokens = word_tokenize(cleaned_text_punc)
    stems = [stemmer.stem(word) for word in word_tokens] 
    stemmed_word_sent = ' '.join(stems)
    
    print(stemmed_word_sent)
    x_test = tokenizer.texts_to_sequences([stemmed_word_sent])
    # print(x_test)
    test_padded = pad_sequences(x_test, padding='post', maxlen=max_length)
    # print(x_test1)
    # print(test_padded1)
    prediction = model.predict(test_padded)
    # Get labels based on probability 1 if p>= 0.5 else 0
    
    if prediction >= 0.5:
        return "positive"
    else:
        return "Negative"


text = st.text_input('Input the sentence, to know the sentiment of it')
if text:
    st.write(Get_sentiment(text))
