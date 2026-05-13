import nltk

# Завантажуємо модель для токенів

nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize, sent_tokenize

text = "Hello. How are you? NLTK is very good and super"
sent =  sent_tokenize(text)
print(sent)

words = word_tokenize(text)
print(words)