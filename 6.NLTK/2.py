import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
words = ["running", 'better', 'cars']
for word in words:
    print(word, "->", lemmatizer.lemmatize(word))

# v - дієслово
print(lemmatizer.lemmatize("running", "v"))
