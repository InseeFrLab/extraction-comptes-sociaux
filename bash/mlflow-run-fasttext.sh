#! /bin/bash
AWS_ACCESS_KEY_ID=`vault kv get -field=ACCESS_KEY onyxia-kv/projet-extraction-tableaux/s3_creds` && export AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=`vault kv get -field=SECRET_KEY onyxia-kv/projet-extraction-tableaux/s3_creds` && export AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN
export MLFLOW_S3_ENDPOINT_URL='https://minio.lab.sspcloud.fr'

# Set MLFLOW_TRACKING_URI environment variable
GET_PODS=`kubectl get pods`

while IFS= read -r line; do
    VAR=`echo "${line}" | sed -n "s/.*mlflow-\([0-9]\+\)-.*/\1/p"`
    if [ -z "$VAR" ]; then
        :
    else
        POD_ID=$VAR
    fi
done <<< "$GET_PODS"

export MLFLOW_TRACKING_URI="https://projet-extraction-tableaux-$POD_ID.user.lab.sspcloud.fr"
export MLFLOW_EXPERIMENT_NAME="page_selection_fasttext"

mlflow run ~/work/extraction-comptes-sociaux/ --entry-point page_selection_fasttext --env-manager=local -P remote_server_uri=$MLFLOW_TRACKING_URI
