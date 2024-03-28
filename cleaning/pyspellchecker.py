# Install the pyspellchecker library if you haven't already
# pip install pyspellchecker

from spellchecker import SpellChecker
from .accent_chars import replace_accented_chars

def psc_correct_spelling(sentence):
    
    sentence = replace_accented_chars(sentence)
    
    # Create a spell checker object
    spell = SpellChecker()

    # Split the sentence into words
    words = sentence.split()

    # Find those words that may be misspelled
    misspelled = spell.unknown(words)

    # Correct the misspelled words
    for word in misspelled:
        # Get the most likely correction
        correction = spell.correction(word)
        # Replace the misspelled word in the sentence
        if correction != None:
            sentence = sentence.replace(word, correction)
    
    return sentence


if __name__ == "__main__":
    # Your sentence with a misspelled word
    sentence = "I lớve to work in ktchen and I like to cớk."

    # Correct the sentence
    corrected_sentence = psc_correct_spelling(sentence)

    print("Original sentence:", sentence)
    print("Corrected sentence:", corrected_sentence)
