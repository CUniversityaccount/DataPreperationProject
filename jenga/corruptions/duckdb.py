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
        

class DuckDBCorruptionWordDeletion(TabularCorruption):
    def transform(self, data : DataFrame) -> DataFrame:
        assert self.column in data.columns

        corrupted_data = data.copy(deep=True)
        corrupted_data['index'] = corrupted_data.index

        exceptions = """-_*@"""
        homos = """˗৭Ȣ𝟕бƼᏎƷᒿlO`ɑЬϲԁе𝚏ɡհіϳ𝒌ⅼｍոорԛⲅѕ𝚝սѵԝ×уᴢ"""
        filter_pattern = homos + """\\-_\\*@"""
        filter_pattern = f"[\\w{filter_pattern}]+"

        return duckdb.sql(f"""
            WITH 
                words AS (
                SELECT 
                    index,
                    flatten(   
                        [   
                            regexp_extract_all(
                                ltrim(w, '{exceptions}' ),
                                '{filter_pattern}'
                            )
                            FOR w in 
                            regexp_split_to_array(reviewText, ' ') 
                        ]
                    ) as word
                    FROM df
                ),
                delete_words AS (
                    SELECT 
                        index,
                        word[CAST(floor(len(word) * random()) AS int)] AS selected_word
                    FROM words
                )
            SELECT 
                CASE WHEN random() < {self.fraction} THEN regexp_replace(d.{self.column}, w.selected_word, '') ELSE {self.column} END AS {self.column},
                d.* EXCLUDE (index, {self.column})
            FROM corrupted_data d
            JOIN delete_words w ON d.index = w.index
            WHERE {self.column} IS NOT NULL
            ORDER BY {self.column}
        """).to_df()


