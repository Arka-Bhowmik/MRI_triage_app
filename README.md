# Deep Learning Breast MRI Triage and Segmentation App

This repository consists of source scripts and final docker container for running a streamlit app for deep learning aided breast MRI triage and segmentation. This app uses pre-trained weights discussed in the repository ([MRI_Triage_Normal](https://github.com/Arka-Bhowmik/mri_triage_normal/)). This app can take single MRI image or batch of MRI image (i.e., in NIFTI Format) to either (a) generate AI score for being completely normal and suspicious exam, or (b) segment breast from thorax for other downstream task. Only testing is possible using this app. For re-training, users must refer to the repository ([see](https://github.com/Arka-Bhowmik/mri_triage_normal/tree/main/training)). The app is user-friendly and allow users to define threshold to personalize the deep learning prediction outcome. However, the deafult/best threshold is τ = 0.33 such that the algorithm triage completely normal exams without missing cancer.

Further information can be obtained by writing to Arka Bhowmik (arkabhowmik@yahoo.co.uk).

<div align="center">
<img src="https://github.com/Arka-Bhowmik/MRI_triage_app/assets/56223140/34227ddc-16e0-4b44-8af9-6c0309d4e99f" width="1200">
</div>
<br><br>

## 🐳  docker image of app (with pre-installed packages and weights)
Users can simply download the docker image and use following steps to run the docker image in local machine:

#### Step I: Download/Install Docker Desktop
Choose appropriate OS/architect, [Docker](https://www.docker.com/products/docker-desktop/)

#### Step II: Download App Docker Image 
[MRI_triage_app](https://drive.google.com/file/d/1N9k4Le-vWJWAuTUiGJM-GX2C4uM8Q1aH/view?usp=sharing) and start docker engine in local machine by running docker desktop

#### Step III: Load the downloaded Docker Image 
Open command prompt(Win) or terminal(Mac/Linux)
```
cd C:\Users\Arka\Downloads                  (window command)
cd /Users/Arka/Downloads/                   (other OS)
docker load -i mri_triage_app.tar.gz
```
*The image will appear in dashboard of Docker desktop after completion of loading.*

#### Step IV: RUN the downloaded Docker Image 
Now, in the command prompt(Win) or terminal(Mac/Linux)
```
docker run -v "/c/Users/Arka/Desktop/image_dataset":/data -p 5000:5000 mri_triage:latest    (window command)
docker run -v "/Users/Arka/Desktop/image_dataset":/data -p 5000:5000 mri_triage:latest      (other OS)
```
*This will provide an url (e.g., "http://0.0.0.0:5000") for the app that can be copied to the browser of local machine.* In the above command, use an appropriate path for -v "/path/" to mount the raw data or image path of the local machine with the docker container that can be accessed inside docker from /data.

#### Step V: Modify Image Path in CSV
The app runs inference for single image or batch of NIFTI images. Batch of images are accepted by the app in the form of a file with extension (.csv or .xlsx) having the absolute image paths ordered in row (*see* input folder for csv headers). The uploaded csv file should have same header to avoid error while the app attempt to save the probabilities.
```
Also modify all "File_path" column in CSV/XLSX during batch run
(e.g., C:/Users/Arka/Desktop/image_dataset/XYZ/abc.nii.gz   to   /data/XYZ/abc.nii.gz)
since docker already mounted /c/Users/Arka/Desktop/image_dataset/    as    /data   in step IV
```
Next, upload the .xlsx or .csv file and run the inference.

#### Step VI: COPY output files
The app saves output "probability.csv" and ROC plot in folder output. The app only generate "roc.png" if the ground truth positive or negative is greater than #15. The saved output files can be copied from docker container from Docker desktop dashboard terminal (*see* [output](https://github.com/Arka-Bhowmik/MRI_triage_app/tree/main/output)). 
```
cp /output/probability.csv /data/
cp /output/roc.png /data/       (only applicable for list of images)
```
This will save the output files in the mounted folder "/Users/Arka/Desktop/image_dataset".


##### Note: For restricted server (*see* [Steps](https://github.com/Arka-Bhowmik/MRI_triage_app/blob/main/tempDir/README.md)).

