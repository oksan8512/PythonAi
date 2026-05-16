import pandas as pd
import re
import string
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ─── Ручна токенізація та стоп-слова (без завантаження NLTK через мережу) ─────
import nltk.corpus
from nltk.corpus import stopwords as nltk_sw

try:
    STOPWORDS = set(nltk_sw.words("english"))
except Exception:
    # Fallback: зчитати вручну створений файл
    with open("/root/nltk_data/corpora/stopwords/english") as f:
        STOPWORDS = set(f.read().splitlines())

def simple_tokenize(text: str) -> list[str]:
    """Проста токенізація: розбиття по пробілах (аналог word_tokenize)."""
    return text.split()

def nltk_bigrams(tokens: list[str]):
    """Генератор біграм (аналог nltk.bigrams)."""
    return zip(tokens, tokens[1:])


# ══════════════════════════════════════════════════════════════
# ЗАВДАННЯ 1: Підготовка та завантаження даних
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("ЗАВДАННЯ 1: Підготовка та завантаження даних")
print("=" * 60)

reviews_data = {
    "id": list(range(1, 16)),
    "review": [
        "The product quality is absolutely amazing! I've been using it for 3 months and it still works perfectly. Highly recommend to everyone!",
        "Terrible experience. The item arrived broken and customer support took 2 weeks to respond. Very disappointed with this purchase.",
        "Good value for money. It does exactly what it says on the box. Shipping was fast and packaging was great. Will buy again!",
        "I ordered the wrong size but the return process was smooth and hassle-free. The replacement arrived within 48 hours. 5 stars!",
        "Average product. Nothing special about it. Works as expected but the build quality feels cheap. Not worth the price in my opinion.",
        "Absolutely love this! It has completely changed my daily routine. The design is sleek and modern. Customer service was helpful too.",
        "Do NOT buy this product. It stopped working after just 1 week. Complete waste of money. The company refused to issue a refund.",
        "Very impressed with the quality and attention to detail. The instructions were clear and setup took only 10 minutes. Great purchase!",
        "Decent product but the delivery was delayed by 2 weeks without any notification. Communication from the seller was very poor.",
        "This exceeded all my expectations! Beautiful design, excellent build quality, and it works flawlessly. Best purchase I made this year!",
        "The product looks different from the photos on the website. Color and size were not as described. Felt misleading and frustrating.",
        "Outstanding customer service! They helped me troubleshoot an issue within minutes. The product itself is solid and reliable.",
        "I've tried many similar products but this one is by far the best. Durable, easy to use, and great value. Would highly recommend!",
        "Not happy with this purchase. The quality has declined compared to previous versions. Packaging was also damaged on arrival.",
        "Simply fantastic! Works exactly as advertised, ships quickly, and the product feels premium. Will definitely order again soon!",
    ],
}

# Зберегти CSV
df = pd.DataFrame(reviews_data)
df.to_csv("reviews.csv", index=False)
print("✅ Збережено 'reviews.csv'")

# Завантажити через pandas
df = pd.read_csv("reviews.csv")
print(f"\n📄 Завантажено рядків : {len(df)}")
print(f"   Колонки           : {list(df.columns)}")

# Очищення
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["cleaned"] = df["review"].apply(clean_text)
df["word_count"] = df["cleaned"].apply(lambda x: len(x.split()))

print(f"\n📊 Кількість рядків            : {len(df)}")
print(f"   Середня довжина відгуку     : {df['word_count'].mean():.2f} слів")
print(f"   Мін / Макс слів             : {df['word_count'].min()} / {df['word_count'].max()}")


# ══════════════════════════════════════════════════════════════
# ЗАВДАННЯ 2: Токенізація і видалення стоп-слів
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ЗАВДАННЯ 2: Токенізація і видалення стоп-слів")
print("=" * 60)

# Об'єднати всі очищені тексти
all_text = " ".join(df["cleaned"].tolist())

# Токенізація (аналог nltk.word_tokenize)
all_tokens_raw = simple_tokenize(all_text)
total_before = len(all_tokens_raw)

# Фільтрація: стоп-слова + токени < 3 символів
filtered_tokens = [
    t for t in all_tokens_raw
    if t not in STOPWORDS and len(t) >= 3
]
total_after = len(filtered_tokens)

print(f"\n🔢 Токенів ДО  очищення  : {total_before}")
print(f"   Токенів ПІСЛЯ очищення : {total_after}")
print(f"   Видалено               : {total_before - total_after} ({(total_before - total_after) / total_before * 100:.1f}%)")
print(f"\n   Перші 15 токенів після фільтрації:")
print("  ", filtered_tokens[:15])


# ══════════════════════════════════════════════════════════════
# ЗАВДАННЯ 3: Частотний словник і візуалізація
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ЗАВДАННЯ 3: Частотний словник і візуалізація")
print("=" * 60)

freq = Counter(filtered_tokens)
top15 = freq.most_common(15)

print(f"\n📋 Топ-15 найчастіших слів:")
print(f"  {'#':<4} {'Слово':<20} {'Частота'}")
print("  " + "-" * 32)
for rank, (word, count) in enumerate(top15, 1):
    bar = "█" * count
    print(f"  {rank:<4} {word:<20} {count:>3}  {bar}")

# ── Горизонтальна діаграма ────────────────────────────────────
words  = [w for w, _ in top15][::-1]
counts = [c for _, c in top15][::-1]

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#0f1117")

# Кольоровий градієнт по частоті
colors = plt.cm.plasma([c / max(counts) for c in counts])
bars = ax.barh(words, counts, color=colors, height=0.65,
               edgecolor="#0f1117", linewidth=0.5)

# Підписи значень на барах
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height() / 2,
            str(count), va="center", ha="left",
            color="white", fontsize=11, fontweight="bold")

ax.set_xlabel("Frequency", color="#cccccc", fontsize=12, labelpad=10)
ax.set_ylabel("Word", color="#cccccc", fontsize=12, labelpad=10)
ax.set_title("Top 15 Frequent Words", color="white",
             fontsize=16, fontweight="bold", pad=18)

ax.tick_params(colors="#cccccc", labelsize=11)
ax.spines[:].set_visible(False)
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.set_xlim(0, max(counts) + 1.5)
ax.grid(axis="x", color="#2a2d36", linewidth=0.7, linestyle="--")

plt.tight_layout()
plt.savefig("feedback_word_freq.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\n✅ Графік збережено: feedback_word_freq.png")


# ══════════════════════════════════════════════════════════════
# ЗАВДАННЯ 4: Аналіз біграм
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ЗАВДАННЯ 4: Аналіз біграм")
print("=" * 60)

# Генерація біграм із відфільтрованих токенів
bigram_list  = list(nltk_bigrams(filtered_tokens))
bigram_freq  = Counter(bigram_list)
top10_bigrams = bigram_freq.most_common(10)

print(f"\n📋 Топ-10 найчастіших біграм:")
print(f"  {'#':<4} {'Біграма':<30} {'Частота'}")
print("  " + "-" * 42)
for rank, (bigram, count) in enumerate(top10_bigrams, 1):
    print(f"  {rank:<4} {' '.join(bigram):<30} {count}")

# Зберегти в CSV
bigram_df = pd.DataFrame(
    [(" ".join(b), c) for b, c in top10_bigrams],
    columns=["bigram", "frequency"]
)
bigram_df.to_csv("feedback_bigrams.csv", index=False)
print("\n✅ Біграми збережено: feedback_bigrams.csv")
print("=" * 60)
print("\n🏁 Всі завдання виконано успішно!")