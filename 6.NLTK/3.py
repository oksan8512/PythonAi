import stanza
stanza.download("uk")

nlp = stanza.Pipeline("uk")

text = """
Ще не вмерла Україна, і слава, і воля,
Ще нам, браття молодії, усміхнеться доля.
Згинуть наші вороженьки, як роса на сонці,
Запануєм і ми, браття, у своїй сторонці.
"""

doc = nlp(text)

for sent in doc.sentences:
    for word in sent.words:
        print(f"{word} -> {word.lemma}")