import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import bigrams

# Завдання 1
reviews = [
    "The product quality is absolutely amazing and exceeded all my expectations!",
    "Delivery was fast but the packaging was damaged. Product itself works great.",
    "Very good quality product. I would highly recommend it to everyone.",
    "The customer service team was incredibly helpful and resolved my issue quickly.",
    "Disappointed with the quality. Expected much better for this price range.",
    "Fantastic product! Works exactly as described. Will definitely buy again.",
    "Good value for money. The build quality feels premium and durable.",
    "Product arrived on time and was well packaged. Very happy with my purchase.",
    "Not satisfied with the battery life. Product stops working after a few hours.",
    "Excellent quality and fast shipping. Customer support was also very responsive.",
    "The product looks great but feels cheap. Not worth the price in my opinion.",
    "Amazing experience from start to finish. Product quality is top notch.",
    "Very poor quality control. My item had visible defects and scratches.",
    "Great product overall. Easy to use and works perfectly every single time.",
    "The size was not accurate as described. However quality seems decent enough.",
]

df = pd.DataFrame({'review': reviews})
df.to_csv('reviews.csv', index=False)
df = pd.read_csv('reviews.csv')

df['clean'] = df['review'].str.lower()
df['clean'] = df['clean'].str.replace(r'[^a-z\s]', '', regex=True)
df['clean'] = df['clean'].str.replace(r'\d+', '', regex=True)
df['clean'] = df['clean'].str.strip()

word_counts = df['clean'].apply(lambda x: len(x.split()))
print(f"Кількість рядків        : {len(df)}")
print(f"Середня довжина відгуку : {word_counts.mean():.2f} слів\n")

# Завдання 2
stop_words = set(stopwords.words('english'))

def tokenize(text):
    return re.findall(r'\b[a-z]+\b', text)

tokens_before = []
for text in df['clean']:
    tokens_before.extend(tokenize(text))

tokens_after = [
    t for t in tokens_before
    if t not in stop_words and len(t) >= 3
]

print(f"Токенів до очищення  : {len(tokens_before)}")
print(f"Токенів після очищення: {len(tokens_after)}\n")

# Завдання 3
freq = Counter(tokens_after)
top15 = freq.most_common(15)

print("Топ-15 найчастіших слів:")
print(f"{'Слово':<18} {'Частота'}")
print("-" * 26)
for word, count in top15:
    print(f"{word:<18} {count}")

words  = [w for w, _ in top15]
counts = [c for _, c in top15]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(words[::-1], counts[::-1], color='#185FA5', alpha=0.82, height=0.65)
ax.set_xlabel('Frequency', fontsize=12)
ax.set_ylabel('Word', fontsize=12)
ax.set_title('Top 15 Frequent Words', fontsize=14, fontweight='bold')
ax.bar_label(bars, padding=4, fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='x', alpha=0.25)
plt.tight_layout()
plt.savefig('feedback_word_freq.png', dpi=150)
plt.close()
print("\nГрафік збережено: feedback_word_freq.png")

# Завдання 4
per_review_tokens = []
for text in df['clean']:
    toks = [w for w in tokenize(text) if w not in stop_words and len(w) >= 3]
    per_review_tokens.append(toks)

all_bigrams = []
for toks in per_review_tokens:
    all_bigrams.extend(list(bigrams(toks)))

bigram_freq = Counter(all_bigrams)
top10 = bigram_freq.most_common(10)

print("\nТоп-10 біграм:")
print(f"{'Біграма':<26} {'Частота'}")
print("-" * 34)
for (a, b), count in top10:
    print(f"{a + ' ' + b:<26} {count}")

bigram_rows = [{'bigram': f"{a} {b}", 'frequency': c} for (a, b), c in top10]
pd.DataFrame(bigram_rows).to_csv('feedback_bigrams.csv', index=False)
print("\nБіграми збережено: feedback_bigrams.csv")