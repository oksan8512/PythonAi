# Стоп слова

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

text = "Ukraine’s glory and freedom have not yet perished,"

words = word_tokenize(text)

stop_words = set(stopwords.words('english'))

filtered_words = [word for word in words if word.lower() not in stop_words]

print(filtered_words)