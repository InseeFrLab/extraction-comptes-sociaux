# Description du projet

Ce projet a pour but d'extraire des informations de tableaux se situant sur des images scannées de comptes sociaux d'entreprise.

## Mise en route

Pour mettre en place l'environnement nécessaire au lancement du code, lancer à la racine du projet la commande
```./setup.sh```

L'entraînement du modèle de segmentation qui retourne pour une image donnée en entrée un masque donnant la position des tableaux sur cette image et un autre masque donnant la position des colonnes se lance ensuite avec la commande
```python train.py```
avec les options suivantes :
- `--gpus` indique le nombre de GPUs utilisés pour l'entraînement. Par défaut égal à 1, si fixé à 0 l'entraînement se fera sans GPU. Pour le moment seuls 0 et 1 sont supportés ;
- `--s3` indique si les logs doivent être sauvegardés sur MinIO ou localement, par défaut ils sont sauvegardés localement ;
- `--lr` indique la learning rate initiale pour l'entraînement. Sa valeur par défaut est 0.001. 

## Utilisation de Tensorboard pour visualiser les logs

Pour utiliser Tensorboard on peut utiliser le service Tensorflow du DataLab (dans la configuration du service il faut activer Tensorboard dans l'onglet Service).

On peut ensuite lancer Tensorboard avec la commande
```AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY AWS_REGION=us-east-1 S3_ENDPOINT=$AWS_S3_ENDPOINT S3_USE_HTTPS=0 S3_VERIFY_SSL=0 tensorboard --logdir s3://projet-ssplab/comptes-sociaux/logs --host 0.0.0.0```
