import pandas as pd

# Load your dataset
df = pd.read_csv('C:\Users\dilar\Desktop\booksummaries\booksummaries.txt')


# Preprocess text data (cleaning, tokenization, etc.)
# You can use techniques like lowercasing, removing stopwords, and stemming/lemmatization
# For example, using NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    words = nltk.word_tokenize(text.lower())
    words = [ps.stem(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

df['clean_summary'] = df['summary'].apply(preprocess_text)



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['clean_summary'])


user_query = "user's input query"
processed_query = preprocess_text(user_query)
query_vector = vectorizer.transform([processed_query])


from sklearn.metrics.pairwise import cosine_similarity

similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
most_similar_book_index = similarity_scores.argmax()

recommended_book_summary = df['summary'][most_similar_book_index]


import streamlit as st

st.title("Book Recommendation System")

user_query = st.text_input("Enter your query:")
if user_query:
    processed_query = preprocess_text(user_query)
    query_vector = vectorizer.transform([processed_query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)
    most_similar_book_index = similarity_scores.argmax()
    recommended_book_summary = df['summary'][most_similar_book_index]

    st.subheader("Recommended Book Summary:")
    st.write(recommended_book_summary)


