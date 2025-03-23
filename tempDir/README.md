*Note*: This is a temporary folder required by the streamlit app to copy temp files while executing the app. After the task completion, these temp files are deleted.


# HELP (RUN THIS APP ON SECURE SERVER FROM TERMINAL)

## Step 1: SSH to server (open terminal)
```
ssh -L 2222:server_address:2222 -L 2858:server_address:2858 user_name@server_address
```
Here, replace keyword *server_address* with "actual address" and user_name with "actual user login details"

#### Step 2: Copy Github Repository (via terminal)
```
cd /data/Arka/CNN_code/
git clone https://github.com/Arka-Bhowmik/MRI_triage_app.git
```
#### Step 3: Create a conda environment (via terminal) 
```
# Install miniconda
mkdir /data/Arka/myenv
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /data/Arka/myenv/miniconda.sh
bash /data/Arka/myenv/miniconda.sh -b -u -p /data/Arka/myenv/miniconda3
/data/Arka/myenv/miniconda3/bin/conda init bash
rm -r /data/Arka/myenv/miniconda.sh

# Create a conda environment
cd /data/Arka/myenv/
conda create -p /data/Arka/myenv/gpu_env python=3.9.17
```
Here, we installed and created a conda environment called "gpu_env".

#### Step 4: Load the conda environment and set current dir to github folder 
```
conda activate /data/Arka/myenv/gpu_env
cd /data/Arka/CNN_code/MRI_triage_app/
```

#### Step 5: Install neccessary packages inside environment 
```
pip install --upgrade pip
pip install -r requirements.txt
pip install gdown==5.2.0
pip cache purge
```
This will set up neccessary packages.

#### Step 6: Download pre-trained model weights in output folder 
```
cd /data/Arka/CNN_code/MRI_triage_app/output/

# weight for u-net
gdown 1BY45-DKsk3cLqfcU6o0rJpG8QKozUUxg

# weight for 1-5 fold classifier
gdown 1BfA-woTZsDWmvbWep_FWetCJK0CnbPXF
gdown 1Bkzu9PMuSmXWYulcVgkJPFGHmz0KWQlq
gdown 1Bs3eAVmOXfuLgPae0PQJduHVJ2QBICyX
gdown 1BtPFdoxcGhGpSird5Yhp31rO4SwTIjct
gdown 1By51-ch1S3238AFQbOk_HAk1x7WJCqJ3
```
#### Step 7: Copy the raw data folder to server folder (by manual or secure copy means)
```
scp -r -P 22 /Users/Arka/Desktop/image_dataset user_name@server_address:/data/Arka/CNN_code/MRI_triage_app/
```
This will copy the local raw dataset folder "image_dataset" to above specified folder mri_triage_app

#### Step 8: Modify the image path in local drive .csv file
This will ensure when we upload the .csv file using our app to server it will take the "File_path" inside our server
![CSV_screenshot](https://github.com/Arka-Bhowmik/MRI_triage_app/assets/56223140/5f8c7392-5cb5-4e8b-8efb-188beb749cb1)

#### Step 9: Run the streamlit App (via server terminal)
```
streamlit run /data/Arka/CNN_code/MRI_triage_app/testing/app.py --server.port=2858 --server.address=0.0.0.0
```
*Ensure to use the server port initial set (i.e., step 1) to comunicate with local device.* This will generate a url (http://0.0.0.0:2858) that can be accessed by local device by opening in a browser.

#### Step 10: Press predict button (for DL prediction)
![Predict](https://github.com/Arka-Bhowmik/MRI_triage_app/assets/56223140/c32a0841-6f29-4f20-97aa-86c0e64c3a16)
