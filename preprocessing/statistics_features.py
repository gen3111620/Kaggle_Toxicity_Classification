import re

def statistics_upper_words(text):
    upper_count = 0
    for token in text.split():
        if re.search(r'[A-Z]', token):
            upper_count += 1
    return upper_count

def statistics_unique_words(text):
    words_set = set()

    for token in text.split():
        words_set.add(token)

    return len(words_set)

def statistics_characters_nums(text):

    chars_set = set()

    for char in text:
        chars_set.add(char)

    return len(chars_set)

def statistics_swear_words(text, swear_words):
    swear_count = 0
    for swear_word in swear_words:
        if swear_word in text:
            swear_count += 1
    return swear_count