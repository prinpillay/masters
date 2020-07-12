#This is the remote usage file to train on a cloud server
usage () {
  echo "usage: mltrain.sh [local | train | tune] [gs://]job_and_data_dir [path_to/]<input_file>.csv
                  [--data-type ratings|web_views]
                  [--delimiter <delim>]
                  [--use-optimized]
                  [--headers]

Use 'local' to train locally with a local data file, and 'train' and 'tune' to
run on ML Engine.  For ML Engine cloud jobs the data_dir must be prefixed with
gs:// and point to an existing bucket, and the input file must reside on GCS.

Optional args:
  --data-type:      Default to 'ratings', meaning MovieLens ratings from 0-5.
                    Set to 'web_views' for Google Analytics data.
  --delimiter:      CSV delimiter, default to '\t'.
  --use-optimized:  Use optimized hyperparamters, default False.
  --headers:        Default False for 'ratings', True for 'web_views'.
"

}

date

TIME=`date +"%Y%m%d_%H%M%S"`

# change to your preferred region
REGION=us-central1

if [[ $# < 3 ]]; then
  usage
  exit 1
fi

# set job vars
TRAIN_JOB="$1"
BUCKET="$2"
DATA_FILE="$3"
JOB_NAME=wals_ml_${TRAIN_JOB}_${TIME}

# add additional args
shift; shift; shift

if [[ ${TRAIN_JOB} == "local" ]]; then

  ARGS="--train-file $BUCKET/${DATA_FILE} --verbose-logging $@"

  mkdir -p jobs/${JOB_NAME}

  gcloud ai-platform local train \
    --module-name trainer.task \
    --package-path trainer \
    -- \
    --job-dir jobs/${JOB_NAME} \
    ${ARGS}

elif [[ ${TRAIN_JOB} == "train" ]]; then

  ARGS="--gcs-bucket $BUCKET --train-file ${DATA_FILE} --verbose-logging $@"

  gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region $REGION \
    --scale-tier=CUSTOM \
    --job-dir ${BUCKET}/jobs/${JOB_NAME} \
    --module-name trainer.task \
    --package-path trainer \
    --master-machine-type complex_model_m_gpu \
    --config trainer/config/config_train.json \
    --master-machine-type complex_model_m_gpu \
    --runtime-version 1.15 \
    -- \
    ${ARGS}

elif [[ $TRAIN_JOB == "tune" ]]; then

  ARGS="--gcs-bucket $BUCKET --train-file ${DATA_FILE} --verbose-logging $@"

  # set configuration for tuning
  CONFIG_TUNE="trainer/config/config_tune.json"
  for i in $ARGS ; do
    if [[ "$i" == "web_views" ]]; then
      CONFIG_TUNE="trainer/config/config_tune_web.json"
      break
    fi
  done

  gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --scale-tier=CUSTOM \
    --job-dir ${BUCKET}/jobs/${JOB_NAME} \
    --module-name trainer.task \
    --package-path trainer \
    --master-machine-type standard_gpu \
    --config ${CONFIG_TUNE} \
    --master-machine-type complex_model_m_gpu \
    --runtime-version 1.15 \
    -- \
    --hypertune \
    ${ARGS}

else
  usage
fi

date
