import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="extraction_core_comptes",
    version="0.0.1",
    author="Tom Seimandi",
    description="Package pour l'extraction de tableau des comptes sociaux",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/InseeFrLab/extraction-comptes-sociaux",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas",
        "pyyaml",
        "numpy",
        "requests",
        "s3fs",
        "matplotlib",
        "pytorch_lightning==2.0.0",
        "torchvision==0.15.1",
        "albumentations==1.3.0",
        "pytesseract",
        "mlflow",
        "nltk",
        "Pillow",
        "pymupdf",
        "unidecode",
        "fasttext"
    ],
    packages=[
        "ca_extract.page_selection",
        "ca_extract.extraction",
        "ca_extract.extraction.data",
        "ca_extract.extraction.tablenet",
        "ca_extract.extraction.table_transformer",
    ],
    python_requires=">=3.7",
    package_data={"extraction_core_comptes": ["data/*"]},
)
