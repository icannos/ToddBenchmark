import random


def random_uppercase_word(text, p=0.1):
    """Randomly uppercase words in a string with probability p."""
    words = text.split()
    for i, word in enumerate(words):
        if random.random() < p:
            words[i] = word.upper()
    return " ".join(words)


def delete_random_char(text, p=0.1):
    """Randomly delete characters in a string with probability p."""
    chars = list(text)
    for i, char in enumerate(chars):
        if random.random() < p:
            chars[i] = ""
    return "".join(chars)
