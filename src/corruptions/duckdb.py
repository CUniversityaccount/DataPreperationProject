from ..jenga.basis import TabularCorruption 
import duckdb
from pandas import DataFrame
import pandas.api.types as ptypes
class DuckDBCorruptionBrokenCharacters(TabularCorruption):
    def transform(self, data : DataFrame) -> DataFrame:
        assert self.column in data.columns
        return duckdb.sql(
            f"""
            SELECT 
                * REPLACE(
                    CASE WHEN random() < {self.fraction} THEN translate({self.column}, 'aAeEoOuU', 'áÁéÉớỚúÚ')  ELSE {self.column} END AS {self.column}
                )
            FROM data
            ORDER BY {self.column}
            """
        ).to_df()


