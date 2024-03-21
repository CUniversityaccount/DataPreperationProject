from ..jenga.basis import TabularCorruption

from textattack.transformations import Transformation, CompositeTransformation
from textattack.constraints import PreTransformationConstraint
from textattack.augmentation import Augmenter

from pandas import DataFrame
from typing import List
import random

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




