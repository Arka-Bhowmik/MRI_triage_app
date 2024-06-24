*Note*: This is a temporary folder required by the streamlit app to copy temp files while executing the app. After the task completion, these temp files are deleted.


# HELP (RUN THIS APP ON SECURE SERVER FROM TERMINAL)

## Step I: SSH to server (open terminal)
```
ssh -L 2222:server_address:2222 -L 2858:server_address:2858 user_name
```
Here, replace keyword *server_address* with "actual address" and user_name with "actual user login details"

#### Step II: Copy Github Repository (via terminal)
```
cd /data/Arka/CNN_code/
git clone https://github.com/Arka-Bhowmik/MRI_triage_app.git
```
#### Step III: Create a conda environment (via terminal) 
```
# Install miniconda
mkdir /data/Arka/myenv
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /data/Arka/myenv/miniconda.sh
bash /data/Arka/myenv/miniconda.sh -b -u -p /data/Arka/myenv/miniconda3
/data/Arka/myenv/miniconda3/bin/conda init bash
rm -r /data/Arka/myenv/miniconda.sh

# Create a conda environment
cd /Users/Arka/myenv/
conda create -p /Users/Arka/myenv/gpu_env python=3.9.17
```
Here, we installed and created a conda environment.

#### Step IV: Load the conda environment and set current dir to github folder 
```
conda activate /data/Arka/myenv/gpu_env
cd /data/Arka/CNN_code/mri_triage_app/
```

#### Step V: Install neccessary packages inside environment 
```
pip install --upgrade pip
pip install -r requirements.txt
pip cache purge
```
This will set up neccessary packages.

#### Step VI: Download the pre-trained model weights in output folder 


