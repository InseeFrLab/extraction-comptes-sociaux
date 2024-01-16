"""
TableCleaner class.
"""
from typing import List
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
import regex


class TableCleaner:
    """
    Cleaner for the table output by `TableExtractor.extract`.
    """

    def __init__(
        self,
        colnames_out: List[str],
        regexes: List[str],
        pct_digit: float = 0.2,
    ):
        """
        Constructor for the TableCleaner class.

        Args:
            colnames_out (List[str]): Columns desired in the output.
            regexes (List[str]): Regular expressions to find in column names.
            pct_digit (float): Minimum percentage of digits in a row (with
                the first column removed) to have it considered as numeric.
        """
        self.colnames_out = colnames_out
        self.regexes = regexes
        self.pct_digit = pct_digit

    def clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selects relevant columns in the pd.DataFrame given as input and
        returns a filtered pd.DataFrame with a clean index and clean
        column names.

        Args:
            df (pd.DataFrame): DataFrame to clean.
        """
        df_without_first_column = df.iloc[:, 1:]
        row_types = self.get_row_types(df_without_first_column)

        # First row can never be values already
        row_types[0] = False

        header_rows = np.arange(np.where(row_types)[0][0])
        body_rows = np.arange(np.where(row_types)[0][0], df.shape[0])

        header_df = df.iloc[header_rows]
        body_df = df.iloc[body_rows]

        similarities = []
        fuzzy_matches = []

        for name, value in header_df.iteritems():
            value_cleaned = value.dropna()
            content = " ".join(value_cleaned)
            similarities.append(
                np.array(
                    [
                        self.similar(content, field)
                        for field in self.colnames_out
                    ]
                )
            )
            fuzzy_matches.append(
                np.array(
                    [
                        1
                        if regex.search(
                            regex_pattern, content, flags=regex.IGNORECASE
                        )
                        else 0
                        for regex_pattern in self.regexes
                    ]
                )
            )

        similarities = np.stack(similarities)
        fuzzy_matches = np.stack(fuzzy_matches)

        column_match = self.stable_match(similarities * fuzzy_matches)
        scores = np.take_along_axis(
            similarities * fuzzy_matches,
            np.expand_dims(column_match, 0),
            axis=0,
        )

        column_match_with_nan = np.where(scores, column_match, -1).squeeze()
        filtered_colnames_out = [
            i
            for (i, v) in zip(
                self.colnames_out, list(column_match_with_nan >= 0)
            )
            if v
        ]
        df_to_return = body_df.iloc[
            :, column_match_with_nan[column_match_with_nan >= 0]
        ]
        df_to_return.columns = filtered_colnames_out
        df_to_return.index = body_df.iloc[:, 0]
        if df_to_return.shape[1] == 0:
            raise IndexError
        return df_to_return.dropna(axis=0, how="all")

    def get_row_types(self, df: pd.DataFrame) -> List[bool]:
        """
        From an input DataFrame, returns a list of booleans and of
        length the number of rows in the DataFrame. Each element of this
        list is set to True if the corresponding row has a percentage of
        digit characters greater than pct_digit and False otherwise.

        Args:
            df (pd.DataFrame): DataFrame to get row types for.
        """
        row_types = []

        for row_index, row in df.iterrows():
            row_cleaned = row.dropna()
            row_as_string = "".join(row_cleaned)

            numbers = sum(c.isdigit() for c in row_as_string)

            try:
                if numbers / len(row_as_string) > self.pct_digit:
                    row_types.append(True)
                else:
                    row_types.append(False)
            except ZeroDivisionError:
                row_types.append(False)

        return row_types

    @staticmethod
    def stable_match(similarities: np.array) -> np.array:
        """
        From a 2D matrix of similarities between the column names
        desired in the output and the column names in an input matrix,
        returns a stable matching.

        Args:
            similarities (np.array): Similarity matrix.
        """
        order = similarities.argsort(0)
        ncolumns_to_match = order.shape[1]
        ncolumns = order.shape[0]

        FREE = -1
        match = FREE * np.ones(ncolumns_to_match, dtype=np.int_)
        jnext = ncolumns * np.ones(ncolumns_to_match, dtype=np.int_)
        rev_match = FREE * np.ones(ncolumns, dtype=np.int_)

        while np.any(match == FREE):
            i = np.where(match == FREE)[0][0]
            jnext[i] -= 1
            j = order[jnext[i], i]
            if rev_match[j] == FREE:
                rev_match[j], match[i] = i, j
            else:
                if similarities[j, rev_match[j]] < similarities[j, i]:
                    match[rev_match[j]] = FREE
                    rev_match[j], match[i] = i, j

        return match

    @staticmethod
    def similar(a: str, b: str) -> float:
        """
        Returns the Levenshtein similarity between two strings given as inputs.

        Args:
            a (str): First string.
            b (str): Second string.
        """
        return SequenceMatcher(None, a, b).ratio()
