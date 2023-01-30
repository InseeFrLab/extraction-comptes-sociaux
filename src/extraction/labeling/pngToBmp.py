"""Module permettant de mettre au bon format les masques pour l'apprentissage"""

from PIL import Image
import os

img_files = [f for f in os.listdir("data/Marmot_data") if f.endswith(".png")]
for tab in img_files:
    img = Image.open("data/Marmot_data/" + tab)
    img = img.convert("RGBA")
    img.save("data/Marmot_data/" + tab[:-4] + ".bmp")
    os.remove("data/Marmot_data/" + tab)

img_files = [f for f in os.listdir("data/Marmot_data") if f.endswith(".json")]
for tab in img_files:
    tab = tab[:-5] + ".bmp"
    img = Image.open("data/Marmot_data/" + tab)
    img2 = Image.open("data/column_mask/" + tab[:-4] + ".bmp")
    img3 = Image.open("data/table_mask/" + tab[:-4] + ".bmp")
    img2 = img2.resize(img.size)
    img2 = img2.convert(mode="P", colors=8)
    img3 = img3.resize(img.size)
    img3 = img3.convert(mode="P", colors=8)
    img = img.convert("RGBA")

    img2.save("data/column_mask/" + tab[:-4] + ".bmp")
    img3.save("data/table_mask/" + tab[:-4] + ".bmp")

img_files = [f for f in os.listdir("data/Marmot_data") if f.endswith(".json")]
for tab in img_files:
    os.remove("data/Marmot_data/" + tab)
