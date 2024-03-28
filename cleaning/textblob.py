from textblob import TextBlob
from .accent_chars import replace_accented_chars

def tb_correct_spelling(sentence):
    sentence = replace_accented_chars(sentence)
    
    # Create a TextBlob object
    blob = TextBlob(sentence)

    # Correct the spelling
    corrected_sentence = blob.correct()
    
    return corrected_sentence

# Sample sentence with spelling mistakes
sentence = "ktchen"

print("Original sentence:", sentence)
print("Corrected sentence:", tb_correct_spelling(sentence))
