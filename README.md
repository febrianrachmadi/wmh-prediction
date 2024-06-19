# wmh-prediction

Predicting the progression of white matter hyperintensities (WMH) from a single T2-FLAIR brain MRI. This repository is the official implementation of our work on developing deep learning prediction models to predict/estimate the future volume of WMH from a single T2-FLAIR brain MRI. We are writing a new manuscript for this study, but our published works on this topic are as follows.

 - [Prediction of white matter hyperintensities evolution one-year post-stroke from a single-point brain MRI and stroke lesions information (bioRxiv, 2022.12. 14.520239, 2022)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZFo5fiwAAAAJ&sortby=pubdate&citation_for_view=ZFo5fiwAAAAJ:isC4tDSrTZIC)   
 - [Probabilistic Deep Learning with Adversarial Training and Volume Interval Estimation-Better Ways to Perform and Evaluate Predictive Models for White Matter Hyperintensities (International Workshop on PRedictive Intelligence In MEdicine, 168-180, 2021)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZFo5fiwAAAAJ&sortby=pubdate&citation_for_view=ZFo5fiwAAAAJ:-f6ydRqryjwC)
 - [Automatic spatial estimation of white matter hyperintensities evolution in brain MRI using disease evolution predictor deep neural networks (Medical image analysis 63, 101712, 2020)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZFo5fiwAAAAJ&cstart=20&pagesize=80&sortby=pubdate&citation_for_view=ZFo5fiwAAAAJ:aqlVkmm33-oC)

# Installation (tested on the Ubuntu 22.04)

 - Please download a docker image using the following link: https://drive.google.com/file/d/152yBvQAWt-1q6ZxyfsXfSu1IeLhlWLD_/view?usp=drive_link
 - Make sure that a Docker system is installed in your machine, where the GPUs are accessible by Docker's containers (see https://docs.docker.com/get-docker/). 
 - On the Linux's terminal, run  `docker load < wmh-prediction\:0.2.2.tar.gz` (tested on the Ubuntu 22.04).
 - Pull this repository for the tested working folder namely `wmh-prediction`. Tested using the following command on Linux Ubuntu 22.04 `git pull https://github.com/febrianrachmadi/wmh-prediction.git`.
 - Run `bash run-bash.sh` to check all files, including trained weights, are accessible in the initial installation. It should output `FINISHED: Everything looks OK, including GPU is found and can be used.` if everthing is installed correctly.
 - The main program for inference can be run by using the following command `bash predict.sh`. However, it does not know the location of your dataset yet, so it will return something like below. Please read the next section on how to access your dataset.

		> CUDA GPU AVAILABLE: True  
		> brain_dir : /wmh-prediction/data/IO/input/study_1_brain_flair  
		> stroke_dir: /wmh-prediction/data/IO/input/study_1_stroke   
		> brain_list: /wmh-prediction/data/IO/input/study_data_list.csv   
		> sampling_number: 5
		> 
		> Create output folder at: /wmh-prediction/data/IO/input/study_1_output   
		> DONE! Created at: /wmh-prediction/data/IO/input/study_1_output
		> 
		>>>> PERFORMING INFERENCE <<<  
		>>>> SAVE AND DISPLAY RECAP <<<  
		+------+----------+------------+----------+  
		| ID | Min-VE | Point-VE | Max-VE |  
		|------+----------+------------+----------|  
		+------+----------+------------+----------+  
		FINISHED!

TO DO: Publish the Docker image to the Docker Hub. 

# Perform inference using trained models for your dataset

 - To set the path to your dataset, you need to change the variable of `DATA_HOST` inside the bash file of [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh). 
 - After the initial installation and cloning of this repository, the `DATA_HOST=$HOME_HOST"/dataset"` variable located inside the [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh) points to a folder named `./dataset/study_1_brain_flair` which should contain T2-FLAIR brain MRI (without skull), a folder named `./dataset/study_1_stroke` which should contain corresponding masks of stroke lesions, and a CSV file of `./dataset/study_data_list.csv` which should contain a list of data ID that will be processed in the inference (in case you do not want to run inference to all data inside the folders). All MRI data should be in *.nii.gz format. If there are no corresponding masks of stroke lesions found inside the `/study_1_stroke` folder that have the same names in the `./dataset/study_1_brain_flair` folder, it is assumed that there are no stroke lesions in the brain MRI data. Please open the `PUT .nii.gz FILES HERE.txt` files inside the two folders and the `./dataset/study_data_list.csv` file for examples.
 - You can copy your dataset to these two folders of `./dataset/study_1_brain_flair` and `./dataset/study_1_stroke` cloned from this repository, and then list these data in the `study_data_list.csv` file cloned from this repository.
 - If you prefer to not copying data to these folders, you can change the names of the dataset folders containing T2-FLAIR brain MRI and their corresponding stroke lesion masks by changing the variable `DATA_HOST` inside the bash file of [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh). 
 - The output folder's name can be changed via variable `OUTPUT_VOLUME_DIR` inside the bash file of [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh). 

# Changing inference processes (e.g., changing pre-processing, adding post-processing, etc.)

The [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh) will run the compiled version of inference. If you want to change how the inference process work (e.g., adding pre-processing, post-processing, etc.), you can use the bash file of [`predict-dev.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict-dev.sh) which will run [`./wmh-prediction/dev/inference-dev.py`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/dev/inference-dev.py). You can freely edit the [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh) file. The bash file `predict-dev.sh` has similar variables like the [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh) with several modifications.

# Create virtual environment on you machine, instead of a Docker container

Here are list of things that need to be installed on your conda/virtual environment. You can download the trained weights using [`download-weights.py`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/dev/download-weights.py) and use the [`predict-dev.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict-dev.sh)  as your main code to edit.

 - NVIDIA GPUs driver and cuda
 - pip install tensorflow[and-cu da]==2.12.0 (tensorflow with GPU version 2.12 tested just fine)
 - pip install gdown
 - pip install pandas
 - pip install nibabel
 - pip install tabulate
 - pip install matplotlib
 - pip install scikit-learn
 - pip install tensorflow-addons

# Questions?

Please let me know should you have any questions via GitHub. I will make sure to check them regularly..
