# DSDT Meetup GoogleML Keras
ML engine PoC for the DSDT meetup

This repository contains everything to create a model on ml-engine:
- a docker Jupyter Notebook to preprocess the dataset, generate predictions and analyze results
- a docker Tensorflow to follow the trainings
- scripts to run the model locally or on the cloud and support hyperparameter research


## Requirements
### Docker
If not already done, you need to download and install [Docker](https://docs.docker.com/).

### Google Cloud SDK
#### Installation
If not already done, you need to download, install and inititialize the [Google Cloud SDK](https://cloud.google.com/sdk/).
After you installed it, run:
```
gcloud init
```

#### Getting credentials
To run this Notebooks server, you'll need your GCloud credential in a json file.  
To obtain them, just run the following command (need to login in a web page):
```
gcloud auth application-default login 
```
You can check if the file has been correctly generated:
```
ls ~/.config/gcloud/application_default_credentials.json
```
This example needs to have the dataset available in a GCloud bucket.
You can run the following command to switch to your project:
```
gcloud config set project project-name
```

## Preprocessing
You can prepare your Python files to access and pre-process the data needed to train the model.
Launch the Jupyter Notebook server with the following command:
```
./scripts/launch_notebooks.sh
```
Then launch your favorite browser and connect to `http://localhost:8888`
Note that in this case, the dataset (images) will need to be copied locally (in the computer 
or in the container in the Cloud) as the used libraries don't support GCloud buckets.
The python functions of this repository are taking care of it.

## Training
The model needs to be presented as a Python package in order to be uploaded and installed in the Cloud.
You can find the model in `src/python/model`  
The setup.py contains the description, the version and the dependencies to install within the container.
ML-engine containers already have tensorflow 1.10, h5py and numpy, so only few libs are needed.

The code is contained in the subdirectory `cnn` and the main entry point is `trainer`.
Note that we support several parameters. These parameters can be adjusted manually by adding them to
the command line or tested automatically using the hyperparameter research functionality of ML-Engine.

To launch a training:
- 1 training job locally:  
Install the dependencies thanks to the requirements.txt file.
It is advised to try ot locally before trying it in the Cloud as the submission of a job can take several minutes 
to start.
Run the command:
```
./scripts/gcloud_local.sh
```
- 1 training job on CPU only in the Cloud:
```
./scripts/gcloud_submit_job.sh
```
- 1 training job on GPU (K80) in the Cloud:
```
./scripts/gcloud_submit_job_gpu.sh
```
- hyperparameter research training jobs (32 jobs in this example) on GPU (K80) in the Cloud:
```
./scripts/gcloud_submit_job_gpu_hyperparams.sh
```
The parameter research is defined in the file `script/gcloud-config-hyperparams.yml`

## Monitoring training
The model is written with `tensorflow.keras` and a callback has been added to write the logs of the training 
in the GCloud bucket.
Run the following command to launch the Tensorboard server:
```
./scripts/launch_tensorboard.sh
```
Then launch your favorite browser and connect to `http://localhost:6006`
You can see the training curves and filter the jobs thanks to regular expressions.

## Analysis
As for the pre-processing you can launch the Jupyter Notebook server with the following command:
```
./scripts/launch_notebooks.sh
```
Then launch your favorite browser and connect to `http://localhost:8888`
All the Python files used for the model are accessible as the directory `src/python` has been added to the Python 
path of the notebook.
You can then predict locally by copying the selected weights locally and loading the model as illustrated in the 
`reseult_analysis` notebook.
