GCS_BUCKET='meetup-gml-keras'

gcloud ml-engine local train \
    --package-path src/python/model/cnn \
    --module-name cnn.trainer \
    -- \
    --dataset-bucket ${GCS_BUCKET} \
    --job-dir ${PWD}/output "$@"
