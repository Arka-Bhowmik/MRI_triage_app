# Use ubuntu LTS version
FROM ubuntu:20.04 AS builder-image

# avoid stuck build due to user prompt
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y python3.9 python3.9-dev python3.9-venv python3-pip vim zsh tmux less curl wget python3-wheel build-essential && \
	apt-get clean && rm -rf /var/lib/apt/lists/*

# create and activate virtual environment
# using final folder name to avoid path issues with packages
RUN python3.9 -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir wheel
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory to /mri_triage_app
RUN mkdir /home/mri_triage_app
WORKDIR /home/mri_triage_app

# Copy the current directory contents into the container in /mri_triage_app
ADD . /home/mri_triage_app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# activate virtual environment
ENV VIRTUAL_ENV=/home/venv
ENV PATH="/home/venv/bin:$PATH"

# Define environment variable
ENV NAME mri_triage

# This sets the default command for the container to run the app with Streamlit.
CMD ["streamlit", "run", "testing/app.py", "--server.port=5000", "--server.address=0.0.0.0"]