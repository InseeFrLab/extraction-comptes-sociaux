# Extraction d'information des comptes sociaux

Ce projet a pour but d'extraire l'information contenue dans le tableau *filiales et participations* qui se situe dans les comptes sociaux des entreprises.

Il est constitué du package `page_selection` qui implémente les fonctionnalités permettant de sélectionner la page du document des comptes sociaux qui contient le tableau des filiales et participations et du package `extraction` qui implémente les fonctionnalités permettant d'extraction du tableau.

## Mise en route

Pour mettre en place l'environnement nécessaire au lancement du code (dans un service du SSP Cloud par exemple), lancer à la racine du projet la commande `./setup.sh`.

Pour l'interaction avec `https://minio.lab.sspcloud.fr`, il est recommandé utiliser les credentials qui figurent comme secrets `Vault` au sein du projet `projet-extraction-tableaux` du SSP Cloud (la commande `unset AWS_SESSION_TOKEN` est conseillée pour éviter des problèmes liés à l'existence de cette variable d'environnement).

## Sélection de page

Le script pour entraîner le modèle de Random Forest qui sert à faire la sélection de page est `train_random_forest.py`. Il se lance avec la commande `./bash/mlflow-run-rf.sh`.

## Extraction

L'entraînement du modèle de segmentation qui retourne pour une image donnée en entrée un masque donnant la position des tableaux sur cette image et un autre masque donnant la position des colonnes figure dans le script `train.py`, et se lance avec la commande `./bash/mlflow-run-tablenet.sh`. Les fichiers de configuration pour l'entraînement se trouvent dans le répertoire `config/` à la racine du projet.
