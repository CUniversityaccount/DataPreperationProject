import pandas as pd
import gzip
import json
import random
import numpy as np
import string
from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.transformations import WordSwapQWERTY
from textattack.transformations import CompositeTransformation
from textattack.transformations import WordSwapRandomCharacterInsertion
from textattack.transformations import WordSwapRandomCharacterSubstitution
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification
from textattack.augmentation import CheckListAugmenter
from textattack.augmentation import WordNetAugmenter
from textattack.transformations import WordSwapWordNet
from textattack.augmentation import Augmenter

def parse(path):
    g = gzip.open(path, "rb")
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient="index")


df = getDF("data/amazon_reviews/AMAZON_FASHION_5.json.gz")
df = df.drop(["vote", "image"], axis=1)

# 创建一个生成随机错误的方法
def augment_text_column(df, column_name):
    # 定义一个组合变换，包括随机字符删除和QWERTY键盘上的字符交换
    transformation = CompositeTransformation([
        WordSwapRandomCharacterDeletion(),
        WordSwapRandomCharacterInsertion(),
        WordSwapRandomCharacterSubstitution(),
        WordSwapQWERTY()
    ])
    
    # 定义约束条件，避免对同一单词重复修改和修改停用词
    constraints = [RepeatModification(), StopwordModification()]
    
    # 创建增强器，应用上述变换和约束
    augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1)
    
    # 对指定列应用增强器
    df[column_name] = df[column_name].apply(lambda x: augmenter.augment(x)[0] if isinstance(x, str) else x)
    
    return df

# 使用CheckListAugmenter添加一个新的方法
def augment_text_with_checklist(df, column_name):
    # 创建CheckListAugmenter实例
    augmenter = CheckListAugmenter(
        transformations_per_example=1,  # 每个示例应用的变换数量
    )
    
    # 对指定列应用增强器
    df[column_name] = df[column_name].apply(lambda x: augmenter.augment(x)[0] if isinstance(x, str) else x)
    
    return df

#使用同义词替换进行攻击
def augment_text_with_wordnet(df, column_name):
    transformation = WordSwapWordNet()
    augmenter = Augmenter(transformation=transformation)
    
    df[column_name] = df[column_name].apply(lambda x: augmenter.augment(x)[0] if isinstance(x, str) else x)
    
    return df


from textattack.transformations import CompositeTransformation, WordSwapWordNet, WordDeletion
from textattack.augmentation import Augmenter
#使用同义词替换和删除随机单词的复合攻击
def augment_text_with_synonym_and_deletion(df, column_name):
    transformation = CompositeTransformation([
        WordSwapWordNet(),
        WordDeletion()
    ])
    augmenter = Augmenter(transformation=transformation)
    
    df[column_name] = df[column_name].apply(lambda x: augmenter.augment(x)[0] if isinstance(x, str) else x)
    
    return df


# 应用于DataFrame的某一列
df = augment_text_column(df, "reviewText")
#df = augment_text_with_checklist(df, "reviewText")
#df=augment_text_with_synonym_and_deletion(df, "reviewText")

if __name__ == "__main__":
    print(df.head(15))

