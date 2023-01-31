#!/bin/bash
# Installing packages
sudo apt-get update
sudo apt-get upgrade -y
sudo apt-get -yq install software-properties-common
sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel
sudo apt -y install python3-opencv
sudo apt -y install tesseract-ocr
sudo apt -y install tesseract-ocr-fra
sudo apt -y install poppler-utils
sudo apt -y install ghostscript python3-tk

# Installing Python requirements
pip install -r requirements.txt

# Install pre-commit
pre-commit install

# Importing data
export MC_HOST_minio=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY:$AWS_SESSION_TOKEN@$AWS_S3_ENDPOINT
mc cp -r minio/projet-extraction-tableaux/data/marmot_data data/
mc cp -r minio/projet-extraction-tableaux/data/table_mask data/
mc cp -r minio/projet-extraction-tableaux/data/column_mask data/
mc cp -r minio/projet-extraction-tableaux/pdf/ data/
mc cp -r minio/projet-extraction-tableaux/raw-comptes/CS_extrait/ data/CS_extrait/
mc cp minio/projet-extraction-tableaux/labels.json data/labels.json
mc cp minio/projet-extraction-tableaux/updated_labels_filtered.json data/updated_labels_filtered.json
mkdir data/csv_table

# Downloading stopwords from the Python `nltk` package
python - <<'END_SCRIPT'
import nltk
nltk.download('stopwords')
END_SCRIPT
