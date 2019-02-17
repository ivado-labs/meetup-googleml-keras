DATE=`date '+%Y%m%d_%H%M%S'`
JOB_NAME=meetup_gml_keras_${DATE}
GCS_BUCKET='meetup-gml-keras'
CONFIG_FILE=$1

if [[ -z "$CONFIG_FILE" ]]
then
      CONFIG_FILE=scripts/gcloud-config.yaml
fi

gcloud ml-engine jobs submit training ${JOB_NAME} \
    --stream-logs \
    --config scripts/gcloud-config.yaml \
    --runtime-version 1.10 \
    --job-dir 'gs://'${GCS_BUCKET} \
    --package-path src/python/model/cnn \
    --module-name cnn.trainer \
    --region us-east1 \
    -- \
    --dataset-bucket ${GCS_BUCKET} "$@"
