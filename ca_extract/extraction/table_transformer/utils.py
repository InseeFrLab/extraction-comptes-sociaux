"""
Plot utils.
"""
from typing import List, Iterable
from PIL import Image, ImageDraw, ImageOps
from matplotlib import pyplot as plt
from fitz import Rect
import fitz
import os
import s3fs
import pandas as pd


fs = s3fs.S3FileSystem(
    client_kwargs={"endpoint_url": "https://" + os.environ["AWS_S3_ENDPOINT"]},
    key=os.environ["AWS_ACCESS_KEY_ID"],
    secret=os.environ["AWS_SECRET_ACCESS_KEY"],
)


def get_plot(image: Image, boxes: List[List[int]]):
    """
    Plot an image with bounding boxes.

    Args:
        image (Image): Image.
        boxes (List[List[int]]): Boxes.
    """
    image_copy = image.copy()
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        c1 = (xmin, ymin)
        c2 = (xmax, ymax)
        draw = ImageDraw.Draw(
            image_copy
        )
        draw.rectangle(
            (c1, c2),
            fill=None,
            outline="red",
            width=3
        )

    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(image_copy)

    return plt.gcf()


def pad_image(image: Image, dilation_factor: float = 0.02) -> Image:
    """
    Pad image.
    """
    color = "white"
    width, height = image.size
    border = (
        int(dilation_factor * height),
        int(dilation_factor * width),
        int(dilation_factor * height),
        int(dilation_factor * width),
    )

    return ImageOps.expand(image, border=border, fill=color)


def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)

    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area

    return 0


def load_pdf(pdf_path: str, s3: bool = True) -> fitz.Document:
    """
    Load pdf file from path.

    Args:
        pdf_path (str): Path to PDF file.
        s3 (bool): True if file is on s3.
    """
    if s3:
        # TODO : clean up
        fs.get(pdf_path, "tmp_pdf.pdf")
        doc = fitz.open("tmp_pdf.pdf")
        os.remove("tmp_pdf.pdf")
    else:
        doc = fitz.open(pdf_path)

    return doc


def get_row_header_ids(df: pd.DataFrame) -> List[int]:
    """
    Get indices of columns corresponding to the row header
    (there can be more than one especially for an
    imperfect extraction).

    Args:
        df (pd.DataFrame): DataFrame.

    Returns
        List[int]: Indices of columns corresponding to the row header.
    """
    # First column is part of the row header
    row_header_ids = [0]

    # Following columns: if entire column is not numeric
    # then it is part of the row header
    specific_pattern = r"^[0-9\(\)\-.,'’` %]*\d[0-9\(\)\-.,'’` %]*$"
    for col_idx, (_, col) in enumerate(df.items()):
        if col_idx == 0:
            continue
        numeric = col.dropna().astype(str).str.contains(specific_pattern)
        if not numeric.any():
            row_header_ids.append(col_idx)
        else:
            break
    return row_header_ids


def get_column_header_ids(df: pd.DataFrame) -> List[int]:
    """
    Get indices of rows corresponding to the column header.

    Args:
        df (pd.DataFrame): DataFrame.

    Returns
        List[int]: Indices of columns corresponding to the column header.
    """
    threshold = 0.5
    specific_pattern = r"^[0-9\(\)\-.,'’ %]*$"
    numeric_cell_rates = []
    for _, row in df.iterrows():
        numeric = row.dropna().astype(str).str.contains(specific_pattern)
        numeric_cell_rates.append(numeric.mean())
    # The last header id is taken as the index before
    # the first numeric row, a.k.a the first row with a
    # mean numeric rate of more than 50% (? - maybe this
    # parameter could be changed)
    numeric_rows = [rate > threshold for rate in numeric_cell_rates]
    index = None
    # First True element
    for i, el in enumerate(numeric_rows):
        if el:
            index = i
            break
    # If first row is numeric
    if index == 0:
        # If only one row is numeric, then it is the header
        # and df is empty, TODO: fix ?
        if len(numeric_rows) == 1:
            return [0]
        # Look at second row and check if numeric :
        # if not then look for the first True element
        # when excluding first row
        elif not numeric_rows[1]:
            for j, el in enumerate(numeric_rows[1:]):
                if el:
                    index = j + 1
                    break
        # Otherwise make first row the header
        else:
            return [0]
    # If no numeric rows ?
    # First heuristic, look for the first row with many empty cells
    elif index is None:
        empty_cell_rates = []
        for _, row in df.iterrows():
            # Check for empty strings or NaN values
            nan_or_empty_string = row.isna() | (row == "")
            empty_cell_rates.append(nan_or_empty_string.mean())
        empty_rows = [rate > threshold for rate in empty_cell_rates]
        # First True element
        for i, el in enumerate(empty_rows):
            if el:
                index = i
                break
        if index == 0:
            # If only numeric rows
            # Edge case we return [0]
            if sum(empty_rows) == len(empty_rows):
                return [0]
            # Look for first non-empty row
            index_first_non_empty = empty_rows.index(False)
            # Starting at that index, if there are only
            # non-empty rows following, first of these is header
            non_all_empty_rows = empty_rows[index_first_non_empty:]
            if sum(non_all_empty_rows) == 0:
                return list(range(index_first_non_empty))
            # Otherwhise we look for the first non-empty row...
            else:
                for i, el in enumerate(non_all_empty_rows):
                    if el:
                        index = i + index_first_non_empty
                        break
    # If still no value for index - maybe not possible at this
    # point who knows
    if index is None:
        return [0]
    # Return list from 0 to last header row
    return list(range(index))


def row_header_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Move columns corresponding to the row header to index.

    Args:
        df (pd.DataFrame): Original df.
    Returns:
        pd.DataFrame: Modified df.
    """
    df_copy = df.copy()
    # Remove missing values in row header
    row_header_ids = get_row_header_ids(df)
    df_copy.iloc[:, row_header_ids] = (
        df_copy.iloc[:, row_header_ids].fillna("").astype(str)
    )
    # Make row header index
    df_copy.index = df_copy.iloc[:, row_header_ids].apply(" ".join, axis=1)
    # Drop columns
    df_copy = df_copy.drop(df_copy.columns[row_header_ids], axis=1)
    return df_copy


def remove_dups_from_list(iterable: Iterable) -> List:
    """
    Remove duplicates from a list while preserving order.

    Args:
        iterable (List): List.

    Returns:
        List: List without duplicates.
    """
    seen = set()
    seen_add = seen.add
    return [x for x in iterable if not (x in seen or seen_add(x))]


def column_header_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Move rows corresponding to the column header to column names.

    Args:
        df (pd.DataFrame): Original df.
    Returns:
        pd.DataFrame: Modified df.
    """
    df_copy = df.copy()
    # Remove missing values in column header
    column_header_ids = get_column_header_ids(df)
    df_copy.iloc[column_header_ids] = (
        df_copy.iloc[column_header_ids].fillna("").astype(str)
    )
    # Make column header column names
    df_copy.columns = df_copy.iloc[column_header_ids].apply(
        lambda x: " ".join(remove_dups_from_list(x)),
        axis=0
    )
    # Drop rows
    df_copy = df_copy.drop(df_copy.index[column_header_ids], axis=0)
    return df_copy


def drop_na_or_identical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with only identical cells and/or NA.

    Args:
        df (pd.DataFrame): Original df.
    Returns:
        pd.DataFrame: Modified df.
    """
    # Drop rows with only identical values
    for _, row in df.iterrows():
        if row.nunique(dropna=False) == 1:
            df = df.drop(index=row.name)
    return df


def drop_empty_rows_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop empty rows and columns.

    Args:
        df (pd.DataFrame): Original df.

    Returns:
        pd.DataFrame: Modified df.
    """
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    return df


def set_index_and_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Set index and column names.

    Args:
        df (pd.DataFrame): Original df.

    Returns:
        pd.DataFrame: Modified df.
    """
    # Order is important ?
    if (df.shape[0] == 0) | (df.shape[1] == 0):
        return df
    df = row_header_to_index(df)
    df = column_header_to_columns(df)
    return df


def format_df_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format df representing a table for comparison.

    Args:
        df (pd.DataFrame): Original df.

    Returns:
        pd.DataFrame: Modified df.
    """
    df = drop_empty_rows_columns(df)
    df = drop_na_or_identical(df)
    df = set_index_and_column_names(df)
    return df
