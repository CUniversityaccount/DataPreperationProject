from transformers import BertTokenizer, BertForMaskedLM
import torch
from spellchecker import SpellChecker

# Function to predict the masked word using BERT
def predict_masked_word(masked_sentence, model, tokenizer, device='cpu'):
    # Tokenize input
    tokenized_input = tokenizer.encode(masked_sentence, return_tensors='pt').to(device)
    # Mask token index
    mask_token_index = torch.where(tokenized_input == tokenizer.mask_token_id)[1]

    # Get predictions for masks
    with torch.no_grad():
        output = model(tokenized_input)
    prediction_scores = output[0]
    
    # Get the predicted token id and convert it to the token string
    predicted_token_id = prediction_scores[0, mask_token_index, :].argmax(axis=1)
    predicted_token = tokenizer.decode(predicted_token_id)

    return predicted_token

def mlm_correct_spelling(sentence):
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertForMaskedLM.from_pretrained('bert-base-cased')

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize pyspellchecker
    spell = SpellChecker()

    # Identify misspelled words
    misspelled_words = spell.unknown(sentence.split())

    # Correct the sentence
    corrected_sentence = sentence
    for misspelled_word in misspelled_words:
        # Create a masked sentence by replacing the misspelled word with the [MASK] token
        masked_sentence = corrected_sentence.replace(misspelled_word, tokenizer.mask_token)
        
        # Predict the correct spelling for the masked word
        corrected_word = predict_masked_word(masked_sentence, model, tokenizer, device)
        
        # Replace the misspelled word with the corrected word
        corrected_sentence = corrected_sentence.replace(misspelled_word, corrected_word)

if __name__ == "__main__":
    # Your sentence with a misspelled word
    sentence = "I love to work in ktchen and I like to cok."

    # Correct the sentence
    corrected_sentence = mlm_correct_spelling(sentence)
    print("Original sentence:", sentence)
    print("Corrected sentence:", corrected_sentence)
