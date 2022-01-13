docker build -t hpc .
gcloud config set project charl-computing
gcloud builds submit --tag gcr.io/charl-computing/hpc

gcloud compute instances create-with-container instance-1 --project=charl-computing --zone=us-central1-a --machine-type=e2-medium --service-account=1092267411362-compute@developer.gserviceaccount.com --container-image=gcr.io/charl-computing/hpc --container-tty --container-restart-policy never
