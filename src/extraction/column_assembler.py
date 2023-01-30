from typing import List
import pandas as pd
import numpy as np


class ColumnAssembler:
    """
    Column Assembler class.
    """

    def __init__(self, with_loc: str = True, threshold_delta: int = 3):
        """
        Constructor for the ColumnAssembler class.

        Args:
            with_loc (str, optional): Indicates if column data
                contains coordinates to help alignment. Defaults to True.
            threshold_delta (int, optional): Value by which to divide
                the column-wise minimum of the mean difference in row
                y coordinates to get the threshold under which one can
                consider that two rows belong to the same cell.
        """
        self.with_loc = with_loc
        self.threshold_delta = threshold_delta

    def assemble(self, column_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Concatenate column DataFrames using coordinates.

        Args:
            column_list (List[pd.DataFrame]): List of column DataFrames.
        """
        if not self.with_loc:
            raise NotImplementedError(
                "Can not concatenate columns without y coordinates."
            )

        mean_top_diffs = [
            (df.top.unique()[1:] - df.top.unique()[:-1]).mean()
            for df in column_list
        ]
        threshold = min(mean_top_diffs) / self.threshold_delta

        for i, df in enumerate(column_list):
            df["df_index"] = i
        all_cols = pd.concat(column_list)
        all_cols = all_cols.sort_values(by="top").reset_index(drop=True)

        diffs = [
            a - b
            for a, b in zip(
                all_cols.top.tolist(), ([0] + all_cols.top.tolist()[:-1])
            )
        ]
        all_cols["diffs"] = diffs
        all_cols["new_line"] = np.where(all_cols.diffs > threshold, 1, 0)
        all_cols["line_id"] = all_cols.new_line.cumsum()
        formatted_cols = [
            all_cols.loc[all_cols.df_index == i]
            for i in range(len(column_list))
        ]
        formatted_cols = [
            formatted_col[["col", "line_id"]].set_index("line_id")
            for formatted_col in formatted_cols
        ]
        formatted_cols = [
            formatted_col.groupby("line_id").agg({"col": " ".join})
            for formatted_col in formatted_cols
        ]

        return pd.concat(formatted_cols, axis=1)
