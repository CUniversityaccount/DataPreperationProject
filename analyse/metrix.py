import spacy
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer

import torch
#Ensure that the corresponding model is installed
#python -m spacy download en_core_web_md
word_vector_similarity= spacy.load('en_core_web_md')  # 确保安装了相应的模型


sentiment_pipeline = pipeline("sentiment-analysis")
nli_model = pipeline('text-classification', model='facebook/bart-large-mnli')

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')



#calculate the similarity between two texts
#This is not a good way to judge the results of a correction
#for example “I love you” and “I hate you” have a high similarity 0.9682823777355798

def get_similarity(text1, text2):
    doc1 = word_vector_similarity(text1)
    doc2 = word_vector_similarity(text2)
    similarity = doc1.similarity(doc2)
    return similarity



#calculate emotional similarity. return 1 if the two texts have the same emotion, otherwise return 0
def get_emotional_similarity(text1,text2):
    result1 = sentiment_pipeline(text1)
    result2 = sentiment_pipeline(text2)
    if result1[0]['label'] == result2[0]['label']:
        return 1 #consistent
    else:
        return 0 #inconsistent


#calculate the cosine similarity between two texts
def get_cosine_similarity(text1,text2):
    inputs = tokenizer([text1, text2], padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    cosine_similarity = torch.nn.CosineSimilarity(dim=0)
    similarity = cosine_similarity(embeddings[0], embeddings[1]).item()
    return similarity

#calculate the logical relationship between two texts
def get_logical_relation(text1, text2):
    result = nli_model(f"{text1} [SEP] {text2}")
    print(result[0]['label'])

