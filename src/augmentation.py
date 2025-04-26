import random
import stanza
import pyiwn
from pyiwn import IndoWordNet, Language

random.seed(42)

iwn = pyiwn.IndoWordNet(lang=Language.HINDI)

nlp = None


def init_stanza():
    global nlp
    if nlp is None:
        try:
            nlp = stanza.Pipeline("hi")
        except:
            stanza.download("hi")
            nlp = stanza.Pipeline("hi")
    return nlp


def synonym_replacement(text, n_versions, p=0.4):
    nlp = init_stanza()
    doc = nlp(text)
    words = [word.text for sent in doc.sentences for word in sent.words]
    pos_tags = [word.upos for sent in doc.sentences for word in sent.words]

    augmented_texts = set()
    max_attempts = n_versions * 2
    attempts = 0

    while len(augmented_texts) < n_versions and attempts < max_attempts:
        new_words = []

        for word, pos in zip(words, pos_tags):
            if pos in ["NOUN", "VERB", "ADJ"] and random.random() < p:
                try:
                    synsets = iwn.synsets(word)
                    if synsets:
                        chosen_synset = random.choice(synsets)
                        synonyms = [
                            lemma.name()
                            for lemma in chosen_synset.lemmas()
                            if lemma.name() != word
                        ]
                        if synonyms:
                            new_words.append(random.choice(synonyms))
                        else:
                            new_words.append(word)
                    else:
                        new_words.append(word)
                except Exception:
                    new_words.append(word)
            else:
                new_words.append(word)

        augmented_text = " ".join(new_words)
        augmented_texts.add(augmented_text)
        attempts += 1

    augmented_list = list(augmented_texts)

    if len(augmented_list) < n_versions:
        augmented_list.extend([text] * (n_versions - len(augmented_list)))

    return augmented_list[:n_versions]


def augment_dataset(df, label_ratios=None):
    if label_ratios is None:
        label_ratios = {1: 21, 0: 6}

    augmented_data = []

    for _, row in df.iterrows():
        text = row["text"]
        label = row["label"]
        n_versions = label_ratios.get(label, 1)
        augmented_texts = synonym_replacement(text, n_versions, p=0.4)
        for aug_text in augmented_texts:
            augmented_data.append({"text": aug_text, "label": label})
        augmented_data.append({"text": text, "label": label})

    import pandas as pd

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df = augmented_df.sample(frac=1, random_state=42).reset_index(drop=True)
    augmented_df = augmented_df.drop_duplicates(subset="text", keep="first")

    return augmented_df