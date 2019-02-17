#!/bin/bash
GCS_BUCKET='meetup-gml-keras'

LOG_PATH_TB='gs://'${GCS_BUCKET} docker-compose -f scripts/docker-compose.yml run --rm --service-ports tensorboard