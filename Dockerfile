FROM tensorflow/tensorflow:2.5.1-gpu

# Any working directory can be chosen as per choice like '/' or '/home' etc
WORKDIR /wmh-prediction/src

# Copy all files/folders at WORKDIR (HOST) to working directory in container
# Now the structure looks like this '/wmh-prediction/src/*'
COPY src ./

# Transfer weights
WORKDIR /wmh-prediction/LBC1936-weights
COPY LBC1936-weights ./
WORKDIR /wmh-prediction/MSS2-weights
COPY MSS2-weights ./

# Go to working directory of /wmh-prediction/
WORKDIR /wmh-prediction
COPY test-installation.py ./
COPY inference.py ./
# COPY download-weights.py ./

RUN pip install gdown
RUN pip install pandas
RUN pip install nibabel
RUN pip install tabulate
RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install tensorflow-addons

# RUN python ./dev/download-weights.py
