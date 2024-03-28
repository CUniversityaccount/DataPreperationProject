def replace_accented_chars(input_str):
    original_chars = 'áÁéÉớỚúÚ'
    replacement_chars = 'aAeEoOuU'
    
    translation_table = str.maketrans(original_chars, replacement_chars)
    return input_str.translate(translation_table)