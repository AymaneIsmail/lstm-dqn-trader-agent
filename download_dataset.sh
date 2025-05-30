#!/bin/bash

KAGGLE_USERNAME="KAGGLE_USERNAME"  
KAGGLE_KEY="KAGGLE_KEY"

DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/pavankrishnanarne/global-stock-market-2008-present"

mkdir -p data

ZIP_PATH="data/global-stock-market-2008-present.zip"

echo "Téléchargement du dataset Kaggle dans $ZIP_PATH ..."

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY -o $ZIP_PATH $DATASET_URL

if [ -f "$ZIP_PATH" ]; then
    echo "Téléchargement terminé, extraction dans ./data"
    unzip -o $ZIP_PATH -d data
else
    echo "Erreur : le téléchargement a échoué."
    exit 1
fi
