import spacy
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import torch

#Ensure that the corresponding model is installed
#python -m spacy download en_core_web_md


#calculate the similarity between two texts
#This is not a good way to judge the results of a correction
#for example “I love you” and “I hate you” have a high similarity 0.9682823777355798

def get_similarity(model, text1, text2):
    doc1 = model(text1)
    doc2 = model(text2)
    similarity = doc1.similarity(doc2)
    return similarity

#calculate emotional similarity. return 1 if the two texts have the same emotion, otherwise return 0
def get_emotional_similarity(model, text1,text2):
    text1 = text1[:512]
    text2 = text2[:512]

    result1 = model(text1)
    result2 = model(text2)
    if result1[0]['label'] == result2[0]['label']:
        return 1 #consistent
    else:
        return 0 #inconsistent

#calculate the cosine similarity between two texts
def get_cosine_similarity(tokenizer, model, text1, text2):    
    inputs1 = tokenizer(text1, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
    inputs2 = tokenizer(text2, padding='max_length', truncation=True, max_length=512, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
    
    embeddings1 = outputs1.last_hidden_state.mean(dim=1)
    embeddings2 = outputs2.last_hidden_state.mean(dim=1)
    
    cosine_similarity = torch.nn.CosineSimilarity(dim=0)
    similarity = cosine_similarity(embeddings1[0].cpu(), embeddings2[0].cpu()).item()
    return similarity

#calculate the logical relationship between two texts
def get_logical_relation(tokenizer, model, text1, text2):
    text1 = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text1)[:509])
    text2 = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text2)[:509])
    result = model(f"{text1} [SEP] {text2}")
    return result[0]['label']

def get_similarity_score(df1, df2, word_vector_similarity, sentiment_pipeline, tokenizer_nli, nli_model, tokenizer_bert, bert_model, device):
    similarity_sum = 0
    emotional_similarity_sum = 0
    cosine_similarity_sum = 0
    logical_relation_sum = 0
    count = 0
    for text1, text2 in zip(df1['reviewText'], df2['reviewText']):
        text1 = str(text1)
        text2 = str(text2)
        similarity = get_similarity(word_vector_similarity, text1, text2)
        emotional_similarity = get_emotional_similarity(sentiment_pipeline, text1, text2)
        cosine_similarity = get_cosine_similarity(tokenizer_bert, bert_model, text1, text2)
        logical_relation_label = get_logical_relation(tokenizer_nli, nli_model, text1, text2)

        similarity_sum += similarity
        emotional_similarity_sum += emotional_similarity
        cosine_similarity_sum += cosine_similarity
        if logical_relation_label == 'entailment':
            logical_relation_sum += 1
        count += 1

    average_similarity = similarity_sum / count
    average_emotional_similarity = emotional_similarity_sum / count
    average_cosine_similarity = cosine_similarity_sum / count
    average_logical_relation = logical_relation_sum / count
    # print('count', count)
    
    return average_similarity, average_emotional_similarity, average_cosine_similarity, average_logical_relation


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    word_vector_similarity= spacy.load('en_core_web_md') 

    tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
    bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
    
    sentiment_pipeline = pipeline("sentiment-analysis", model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', device=device)
    
    tokenizer_nli = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')
    nli_model = pipeline('text-classification', model='facebook/bart-large-mnli', device=device)
    
    df1 = pd.read_json('./data/test.json',lines=True)
    df2_name = ['ta_clean_psc_test', 'ta_clean_tb_test', 'dd_clean_psc_test', 'dd_clean_tb_test', 'jg_clean_psc_test', 'jg_clean_tb_test']
    
    for name in df2_name:
        df2 = pd.read_json(f'./data/{name}.json', lines=True)
        average_similarity, average_emotional_similarity, average_cosine_similarity, average_logical_relation = get_similarity_score(df1, df2, word_vector_similarity, sentiment_pipeline, tokenizer_nli, nli_model, tokenizer_bert, bert_model, device)
        # average_similarity, average_emotional_similarity, average_cosine_similarity = get_similarity_score(df1, df2, word_vector_similarity, sentiment_pipeline, tokenizer_nli, nli_model, tokenizer_bert, bert_model, device)

        print('\n')
        print('name: ', name)
        print('average_similarity: ', average_similarity)
        print('average_emotional_similarity: ', average_emotional_similarity)
        print('average_cosine_similarity: ', average_cosine_similarity)
        print('average_logical_relation: ', average_logical_relation)