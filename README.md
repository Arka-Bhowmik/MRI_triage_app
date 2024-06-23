# Deep Learning Breast MRI Triage and Segmentation App

This repository consists of source scripts and final docker container for running a streamlit app for deep learning aided breast MRI triage and segmentation. This app uses pre-trained weights discussed in the repository ([MRI_Triage_Normal](https://github.com/Arka-Bhowmik/mri_triage_normal/)). This app can take single MRI image or batch of MRI image (i.e., in NIFTI Format) to either (a) generate AI score for being completely normal and suspicious exam, or (b) segment breast from thorax for other downstream task. Only testing is possible using this app. For re-training, users must refer to the repository ([see](https://github.com/Arka-Bhowmik/mri_triage_normal/tree/main/training)). The app is user-friendly and allow users to define threshold to personalize the deep learning prediction outcome. However, the deafult/best threshold is τ = 0.33 such that the algorithm triage completely normal exams without missing cancer.

For further information can be obtained by writing to Arka Bhowmik (arkabhowmik@yahoo.co.uk).

## 🐳  docker image of the app (with pre-installed packages and weights)
Users can simply download the docker image and use following steps to run the docker image:

Step I: Download and install docker desktop for your OS/architect, [Docker](https://www.docker.com/products/docker-desktop/)




Note: For restricted server, the docker image need to be created with appropriate ports in dockerfile provided.

