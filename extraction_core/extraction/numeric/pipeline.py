"""Ce fichier permet d'extraire les tableaux des pdf numerique nottament à l'aide du package camelot.
Ce package permet de traiter des pdf numerique et d'y récupérer le texte exact à l'intérieur des tableaux.
Ce qui est donc plus pertinent plutôt que d'utiliser l'OCRisation comme le fait le reste de l'application pour les pdf scannés.

Methods:
    isnotdigit(): Méthode intermédiaire permettant simplement de vérifier qu'une chaîne de caractère ne contient pas de chiffres.

    clean_column(): Méthode permettant de détecter le nom des colonnes dans les tableaux extraits. Ainsi que de nettoyer les tableaux (suppression de lignes vides par exemple)

    matching_text(): Méthode utilisant des pattern regex pour détecter des colonnes voulues. (Supprime toutes les colonnes que l'utilisateur ne veut pas garder)

    pipeline(): Méthode à lancer pour extraire un tableau d'un fichier pdf. Retourne la dataframe du tableau et crée un fichier csv contenant le tableau.
"""

import os

# import camelot
import regex
import numpy as np
from .tablemask import setup_extractor, find_table
import fitz
from ..constants import fields, regexes


def isnotdigit(string):
    """
    Cette fonction sert à vérifier qu'il n'y a pas de nombre dans un string.
    Elle est nottament utilisée pour détecter les noms de colonnes dans
    les dataframe et dans la fonction clean_column()

    Args:
        string (str): Chaîne de caractère que l'on veut tester

    Returns:
        (boolean): Renvoie True s'il n'y a pas de nombre dans la chaîne de caractère
    """
    return not (any(chr.isdigit() for chr in string))


def clean_column(df):
    """
    Cette fonction permet de détecter les noms de colonnes
    qui se retrouvent dans les observations
    des dataframes obtenus grâce à Camelot.
    Elle nettoie aussi les lignes inutiles.

    Args:
        df (dataframe): Dataframe contenant les données du tableaux
                        obtenus grâce à Camelot

    Returns:
        df (dataframe): Dataframe nettoyé
    """
    for i in range(1, df.shape[1]):
        if isnotdigit(df[i][0]):
            is_str = True
            j = 0
            column_name = ""
            while is_str and j < df.shape[0]:
                if isnotdigit(df[i][j]):
                    name = " " + df[i][j]
                    column_name += name
                    df.iloc[j, i] = np.nan
                    j += 1
                else:
                    is_str = False
        else:
            column_name = "Digit column name"

        # Nettoyage du nom des colonnes
        # df[i].apply(lambda x : clean(x)) # On supprime les cellules qui contiennent des noms de colonnes
        column_name = column_name.replace(
            "\n", ""
        )  # On retire les saut à la ligne
        column_name = column_name.replace(
            ",", ""
        )  # Certaines colonnes ont une virgule qui apparaît
        column_name = column_name.strip()  # On retire les espaces inutiles
        df.rename(columns={i: column_name}, inplace=True)

    # Suppression des colones sans nom (Elles gènent lors de l'utilisation de la fonction matching_text)
    i = 0
    while i < df.shape[1]:
        if df.columns[i] == "":
            df.drop(df.columns[i], axis=1, inplace=True)
        elif df.columns[i] == "Digit column name":
            df.drop(df.columns[i], axis=1, inplace=True)
        else:
            i += 1

    # Suppresion des lignes vides
    columns_to_consider = df.columns[1:]
    df = df.dropna(axis=0, how="all", subset=columns_to_consider)

    # # Rectification des colonnes Valeure brutte/valeure nette
    # list_column = [column for column in df]
    # for column_name in list_column:
    #     if column_name == 'Brut' or column_name == 'Brutte' or column_name == 'Brute':
    #         df.rename(columns={column_name: 'Valeur brute comptable des titres détenus'})
    #     elif column_name == 'Net' or column_name == 'Nette':
    #         df.rename(columns={column_name: 'Valeur nette comptable des titres détenus'})
    return df


def matching_text(df, fields, regexes):
    """
    Permet de détecter les colonnes d'intérêt dans la dataframe. Ces colonnes sont spécifiés dans la
    liste fields. La méthode utilise regex pour détecter ces colonnes. Une fois ces colonnes détéctées,
    la méthode supprimera toutes les autres colones de la dataframe.

    Args:
        df (dataframe): Dataframe à filtrer.
        fields (list[str]) : Liste des noms de colonnes d'intérêt à garder.
        regexes (list[str]) : Liste des patterns regex permettant de détecter les colonnes d'intérêt.

    Return:
        df (dataframe): Dataframe filtrée.
    """

    df_without_first_column = df.iloc[:, 1:]

    for column_name in df_without_first_column:
        matching = False
        for i in range(5):
            pattern = regexes[i]
            if regex.search(pattern, column_name) is not None:
                df = df.rename(columns={column_name: fields[i]})
                # df.rename(columns={column_name: fields[i]}, inplace=True)
                matching = True
        if matching == False:
            if column_name in [column for column in df]:
                del df[column_name]
    return df


def pipeline(filename, table_extractor):
    """
    Fonction principale permettant de traiter un fichier pdf afin de parser le tableau et l'exporter
    en csv. Cette méthode utilise les méthodes clean_column et matching text.

    Args:
        filename (str): Nom du fichier pdf à traiter.
    """
    path = (
        "/home/coder/work/comptes-sociaux-to-tableau-fp-csv/extraction_core/data/pdf_numerique/"
        + filename
    )
    print("Le fichier " + str(filename) + " est en train d'être traité.")

    tables = []
    # Toute cette partie propose différentes méthode pour extraire
    # les tableaux, la meilleure reste celle intitulé 'Table Extractor'

    # Flavor Lattice

    # tables1 = camelot.read_pdf(path, line_scale=20)
    # if tables1.n != 0:
    #     for i in range(tables1.n):
    #         tables.append(tables1[i])

    # # Flavor Lattice extreme
    # tables2 = camelot.read_pdf(path, lince_scale=40, process_background=True, line_tol=1.5)
    # if tables2.n != 0:
    #     for i in range(tables2.n):
    #         tables.append(tables2[i])

    # # Flavor Lattice extreme 2
    # tables3 = camelot.read_pdf(path, lince_scale=75, process_background=True)
    # if tables3.n != 0:
    #     for i in range(tables3.n):
    #         tables.append(tables3[i])

    # # Flavor Stream
    # tables4 = camelot.read_pdf(path, flavor='stream')
    # if tables4.n != 0:
    #     for i in range(tables4.n):
    #         tables.append(tables4[i])

    # # Flavor Stream extreme
    # tables5 = camelot.read_pdf(path, flavor='stream', edge_tol=500)
    # if tables5.n != 0:
    #     for i in range(tables5.n):
    #         tables.append(tables5[i])

    # Table Extractor
    try:
        tables6 = find_table(path, table_extractor)
        if len(tables6) != 0:
            for i in range(len(tables6)):
                tables.append(tables6[i])

        list_shape = []
        list_df = []
        for table in tables:
            df = table.df
            df = clean_column(df)
            matching_text(df, fields, regexes)

            # Suppression des tableaux vide
            if df.shape[1] == 1:
                tables.remove(table)
            else:
                list_shape.append(df.shape[1])
                list_df.append(df)

        # Recherche du tableau le plus intéressant
        if list_shape != []:
            index = np.argmax(list_shape)
            better_df = list_df[index]

            csv_name = filename[:-3] + "csv"
            better_df.to_csv(
                "/home/coder/work/comptes-sociaux-to-tableau-fp-csv/extraction_core/data/csv_table/"
                + csv_name,
                index=False,
            )
    except fitz.FileDataError:
        print("Le pdf n'a pas pu être converti en image.")
    print("Traitement du fichier terminé.")


# ZONE D'EXECUTION
if __name__ == "__main__":

    # Récuperation des noms des pdf sous forme d'une liste de chaîne de caractère
    list_file = os.listdir(
        "/home/coder/work/comptes-sociaux-to-tableau-fp-csv/extraction_core/data/pdf_numerique"
    )

    table_extractor = setup_extractor()

    for file_name in list_file:
        pipeline(file_name, table_extractor)

    # path2 = '487772899_f_et_p.pdf'
    # pipeline(path2, table_extractor)
