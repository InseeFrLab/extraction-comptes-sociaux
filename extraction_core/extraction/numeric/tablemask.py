import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

from ..table_extractor import TableExtractor
from PyPDF2 import PdfReader
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, convex_hull_image
import numpy as np
import camelot
import fitz


def table_coordinate(table):
    """Retourne les coordonées x1, x2, y1, y2 du tableau détécté à l'aide d'un masque.

    Args:
        table (array(int)): Masque (sous forme d'un tableau de 0 et de 1)

    Returns:
        coordinate (tuple[int]): Retourne une liste sous la forme (x1, y1, x2, y2)
    """
    first_ones_y = table.argmax(0)
    y1 = first_ones_y[first_ones_y != 0].min()

    last_ones_y = np.flip(table, axis=0).argmax(0)
    y2 = table.shape[0] - last_ones_y[last_ones_y != 0].min()

    first_ones_x = table.argmax(1)
    x1 = first_ones_x[first_ones_x != 0].min()

    last_ones_x = np.flip(table, axis=1).argmax(1)
    x2 = table.shape[1] - last_ones_x[last_ones_x != 0].min()

    return (x1, x2, y1, y2)


def table_reshape(coordinate, table_mask_shape, original_shape):
    """Redimensionne les coordonnées selon les dimensions d'une image.

    Args:
        coordinate (list[int]): Liste des coordonnées du tableau détecté.
        table_mask_shape (tuple[int]) : Dimension du masque détéctant le tableau.
        original_shape (tuple[int]) : Dimension du document original.

    Returns:
        reordered_coordinates (list[int]): Liste des coordonnées redimensionné
    """
    x1, x2, y1, y2 = coordinate[0], coordinate[1], coordinate[2], coordinate[3]

    x1_original = x1 * int(original_shape[1]) // int(table_mask_shape[1])
    x2_original = x2 * int(original_shape[1]) // int(table_mask_shape[1])
    y1_original = y1 * int(original_shape[0]) // int(table_mask_shape[0])
    y2_original = y2 * int(original_shape[0]) // int(table_mask_shape[0])

    y1_original = int(original_shape[0]) - y1_original
    y2_original = int(original_shape[0]) - y2_original

    resized_coordinates = x1_original, x2_original, y1_original, y2_original
    reordered_coordinates = (
        resized_coordinates[0],
        resized_coordinates[2],
        resized_coordinates[1],
        resized_coordinates[3],
    )

    return reordered_coordinates


def clean_mask(table_mask):
    """Nettoie le masque afin qu'il soit plus représentatif d'un tableau. De plus cette méthode permet
    de détecter s'il y a plusieurs tableau dans un masque.

    Args:
        table_mask (array(int)): Tableau contenant le masque
    """
    thresh = threshold_otsu(table_mask)
    bw = closing(table_mask > thresh, square(10))
    cleared = clear_border(bw)
    segmented_mask = label(cleared)
    return segmented_mask


def separating_mask(segmented_mask):
    """Sépare un masque en différent masque pour chaque tableau contenu dans ce dernier.

    Args:
        segmented_mask (array(int)): Array contenant le masque d'un ou plusieurs tableaux.

    Return:
        tables (list[array(int)]): Liste des différents masques
    """
    width, height = segmented_mask.shape
    tables = []
    for i in np.unique(segmented_mask)[1:]:
        table = np.where(segmented_mask == i, 1, 0)
        if table.sum() > height * width * 0.02:
            tables.append(convex_hull_image(table))
    return tables


def setup_extractor():
    """Crée un objet TableExtractor, permettant d'extraire les tableaux."""
    transforms = album.Compose(
        [
            album.Resize(896, 896, always_apply=True),
            album.Normalize(),
            ToTensorV2(),
        ]
    )

    checkpoints = "/projet-extraction-tableaux/marmot_model.ckpt"
    table_extractor = TableExtractor(
        checkpoint_path=checkpoints, transforms=transforms
    )
    return table_extractor


def find_table(pdf_path, table_extractor):
    """Fonction principale se basant sur tout les autres. Elle utilise Camelot pour
    obtenir le tableau sous forme de dataframe.

    Args:
        pdf_path (string): Chemin d'accès au pdf à traiter
        table_extractor (TableExtractor): Extracteur de tableau

    Returns:
        camelot_tables: Une liste d'objet Camelot caractérisant les tableaux qui ont été extrait.
    """
    pdf_file = pdf_path
    doc = fitz.open(pdf_file)
    page = doc.load_page(0)  # number of page
    pix = page.get_pixmap()
    image_path = "/home/coder/work/comptes-sociaux-to-tableau-fp-csv/extraction_core/numeric/image.png"
    pix.save(image_path)

    image = Image.open(image_path)
    out = table_extractor.extract(image, post_processing=False)
    table_mask = out["table_mask"]

    reader = PdfReader(pdf_path)
    pdf_width = reader.pages[0].mediabox[2]
    pdf_height = reader.pages[0].mediabox[3]
    pdf_shape = [pdf_height, pdf_width]  # Dimensions du pdf

    segmented_mask = clean_mask(table_mask)
    tables = separating_mask(segmented_mask)

    camelot_tables = []

    for table in tables:

        x1, x2, y1, y2 = table_coordinate(table)

        coordinate = [x1, x2, y1, y2]
        reordered_coordinates = table_reshape(
            coordinate, table_mask.shape, pdf_shape
        )

        areas = [",".join(str(coord) for coord in reordered_coordinates)]
        try:
            camelot_table = camelot.read_pdf(
                pdf_path, flavor="stream", table_areas=areas
            )
            camelot_tables.append(camelot_table[0])
        except ValueError:
            print("Ce tableau ne peut pas être extrait")
    return camelot_tables
