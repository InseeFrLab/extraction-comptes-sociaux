#! /bin/bash
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
export MLFLOW_EXPERIMENT_NAME="extraction"

mlflow run ~/work/extraction-comptes-sociaux/ --entry-point extraction --env-manager=local -P remote_server_uri=$MLFLOW_TRACKING_URI -P config_path=config/tablenet_config/tablenet_config.yaml
