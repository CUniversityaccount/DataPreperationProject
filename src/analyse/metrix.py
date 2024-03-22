import spacy
from transformers import pipeline

#Ensure that the corresponding model is installed
#python -m spacy download en_core_web_md
nlp = spacy.load('en_core_web_md')  # 确保安装了相应的模型

#calculate the similarity between two texts
#This is not a good way to judge the results of a correction
#for example “I love you” and “I hate you” have a high similarity 0.9682823777355798
def get_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity = doc1.similarity(doc2)
    return similarity

sentiment_pipeline = pipeline("sentiment-analysis")
# 示例句子
sentence1 = "I love it."
sentence2 = "I hate it."

# 对两个句子进行情感分析
result1 = sentiment_pipeline(sentence1)
result2 = sentiment_pipeline(sentence2)

# 打印结果
print(result1)
print(result2)
if result1[0]['label'] == result2[0]['label']:
    print("These sentences have consistent sentiment.")
else:
    print("These sentences have inconsistent sentiment.")