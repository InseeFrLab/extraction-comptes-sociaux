#!/bin/bash
if [ "$EUID" -ne 0 ]
then
    sudo apt-get -y install software-properties-common
    sudo apt-get update
    sudo add-apt-repository ppa:alex-p/tesseract-ocr-devel
    sudo apt-get update
    # sudo apt install python3.7
    sudo apt -y install python3-opencv
    sudo apt -y install tesseract-ocr
    sudo apt -y install tesseract-ocr-fra
else
    add-apt-repository ppa:alex-p/tesseract-ocr-devel
    apt-get update
    # apt install python3.7
    apt install -y python3-opencv
    apt install -y tesseract-ocr
    apt install -y tesseract-ocr-fra
fi

# pip install virtualenv
# virtualenv --python=/usr/bin/python3.7 venv
# source venv/bin/activate

mkdir -p ~/.config/fsspec && touch ~/.config/fsspec/conf.json
echo '{"s3": {"client_kwargs": {"endpoint_url": "https://minio.lab.sspcloud.fr"}}}' > ~/.config/fsspec/conf.json

pip install -r requirements.txt

export MC_HOST_minio=https://$AWS_ACCESS_KEY_ID:$AWS_SECRET_ACCESS_KEY:$AWS_SESSION_TOKEN@$AWS_S3_ENDPOINT

# mc cp -r minio/projet-extraction-tableaux/weights/ weights/
# export TORCH_HOME=weights/ 

mc cp -r minio/projet-extraction-tableaux/data/marmot_data data/
mc cp -r minio/projet-extraction-tableaux/data/table_mask data/
mc cp -r minio/projet-extraction-tableaux/data/column_mask data/
mc cp -r minio/projet-extraction-tableaux/pdf/ data/
mc cp -r minio/projet-extraction-tableaux/raw-comptes/CS_extrait/ data/CS_extrait/
mc cp minio/projet-extraction-tableaux/labels.json data/labels.json
mc cp minio/projet-extraction-tableaux/updated_labels_filtered.json data/updated_labels_filtered.json
mkdir data/csv_table

sudo apt-get upgrade
sudo apt install -y poppler-utils
sudo apt install -y ghostscript python3-tk

python - <<'END_SCRIPT'
import nltk
nltk.download('stopwords')
END_SCRIPT
