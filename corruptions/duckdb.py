from jenga.basis import TabularCorruption 
import duckdb
from pandas import DataFrame

# Insert broken characters through duckdb
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
            WHERE {self.column} IS NOT NULL
            ORDER BY {self.column}
            """
        ).to_df()


