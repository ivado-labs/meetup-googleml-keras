DATE=`date '+%Y%m%d_%H%M%S'`
JOB_NAME=meetup_gml_keras${DATE}
GCS_BUCKET='meetup-gml-keras'
CONFIG_FILE='scripts/gcloud-config.yaml'

gcloud ml-engine jobs submit training ${JOB_NAME} \
    --stream-logs \
    --scale-tier basic_gpu \
    --config ${CONFIG_FILE} \
    --runtime-version 1.10 \
    --job-dir 'gs://'${GCS_BUCKET} \
    --package-path src/python/model/cnn \
    --module-name cnn.trainer \
    --region us-east1 \
    -- \
    --dataset-bucket ${GCS_BUCKET} "$@"
