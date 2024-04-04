from jenga.basis import TabularCorruption

from textattack.transformations import Transformation, CompositeTransformation
from textattack.constraints import PreTransformationConstraint
from textattack.augmentation import Augmenter

import random
import duckdb

from duckdb.typing import VARCHAR
from pandas import DataFrame
from typing import List

class TextAttackCorruption(TabularCorruption):
    '''
    Corruption on text through text attack transformations
    Input:
    column: name of the column to corrupt
    fraction: fraction that is going to be corrupted
    transformations: the possible transformations done on the text
    constraint: constraint of text attack where the augmenter needs to take care off.
    '''
    def __init__(
            self,
            column : str,
            fraction : float,
            transformations : List[Transformation],
            constraints : List[PreTransformationConstraint] = []

    ):
        self.column = column
        self.fraction = fraction
        self.transformations = transformations
        self.constraints = constraints

    def transform(self, data : DataFrame) -> DataFrame:
        assert self.column in data.columns
        assert self.transformations and len(self.transformations) > 0

        corrupted_data = data.copy(deep=True)
        transformator = CompositeTransformation(self.transformations)
        augmenter = Augmenter(
            transformation=transformator,
            constraints=self.constraints,
            pct_words_to_swap=0.1
        )

        for index, row in corrupted_data.iterrows():
            if random.random() >= self.fraction:
                continue;

            column_value = row[self.column]
            augmented_text = augmenter.augment(column_value)[0]
            corrupted_data.at[index, self.column] = augmented_text

        return corrupted_data
    

class TextAttackThroughDuckDBCorruption(TabularCorruption):
    '''
    Corruption on text through text attack transformations
    Input:
    column: name of the column to corrupt
    fraction: fraction that is going to be corrupted
    transformations: the possible transformations done on the text
    constraint: constraint of text attack where the augmenter needs to take care off.
    '''
    def __init__(
            self,
            column : str,
            fraction : float,
            transformations : List[Transformation],
            constraints : List[PreTransformationConstraint] = []

    ):
        self.column = column
        self.fraction = fraction
        self.transformations = transformations
        self.constraints = constraints

    def transform(self, data : DataFrame) -> DataFrame:
        assert self.column in data.columns
        assert self.transformations and len(self.transformations) > 0

        transformator = CompositeTransformation(self.transformations)
        augmenter = Augmenter(
            transformation=transformator,
            constraints=self.constraints,
            pct_words_to_swap=0.1
        )
        def augment(w : str) -> str: return augmenter.augment(w)[0]
        duckdb.create_function("augment", augment)
        df = duckdb.sql(
            f"""
            SELECT 
                * REPLACE(
                    CASE WHEN random() < {self.fraction} THEN augment({self.column})  ELSE {self.column} END AS {self.column}
                )
            FROM data
            WHERE {self.column} IS NOT NULL
            ORDER BY {self.column}
            """
        ).to_df()
        duckdb.remove_function("augment")
        return df




