# wmh-prediction

Predicting the progression of white matter hyperintensities (WMH) from a single T2-FLAIR brain MRI. This is the official implementation of our works on developing deep learning prediction models to predict/estimate the future volume of WMH from a single T2-FLAIR brain MRI. Our published works on this topic are as follows. 

 - [Prediction of white matter hyperintensities evolution one-year post-stroke from a single-point brain MRI and stroke lesions information (bioRxiv, 2022.12. 14.520239, 2022)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZFo5fiwAAAAJ&sortby=pubdate&citation_for_view=ZFo5fiwAAAAJ:isC4tDSrTZIC)   
 - [Probabilistic Deep Learning with Adversarial Training and Volume Interval Estimation-Better Ways to Perform and Evaluate Predictive Models for White Matter Hyperintensities (International Workshop on PRedictive Intelligence In MEdicine, 168-180, 2021)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZFo5fiwAAAAJ&sortby=pubdate&citation_for_view=ZFo5fiwAAAAJ:-f6ydRqryjwC)
 - [Automatic spatial estimation of white matter hyperintensities evolution in brain MRI using disease evolution predictor deep neural networks (Medical image analysis 63, 101712, 2020)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=ZFo5fiwAAAAJ&cstart=20&pagesize=80&sortby=pubdate&citation_for_view=ZFo5fiwAAAAJ:aqlVkmm33-oC)

We are in the middle of writing a new manuscript for this particular study.

# Installation
TO DO: Publish the Docker image to the Docker Hub?
For now, please download a docker image from the following link.

 - [ ] https://drive.google.com/file/d/152yBvQAWt-1q6ZxyfsXfSu1IeLhlWLD_/view?usp=drive_link
 - [ ] Make sure that you have Docker where GPUs must be able to accessed by Docker's containers (see https://docs.docker.com/get-docker/).  Tested on the Ubuntu 22.04.
 - [ ] On Linux's terminal, run  `docker load < wmh-prediction\:0.2.2.tar.gz`. Tested on the Ubuntu 22.04.
 - [ ] Pull this repository for the tested working folder named `wmh-prediction`. Tested using the following command on Linux Ubuntu 22.04 `git pull https://github.com/febrianrachm  
adi/wmh-prediction.git`.
 - [ ] Run `bash run-bash.sh` to check all need files, including trained weights, are accessible in the initial installations. It should output `FINISHED: Everything looks OK, including GPU is found and can be used.` if everthing is OK.
 - [ ] The main program for inference can be run by using the following command `bash predict.sh`. However, it still does not know where to look your dataset so it will return something like below. Please read the next section to access your own dataset.

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

# Perform inference using trained models for your dataset

To set the path to your dataset, you need to change the variable of `DATA_HOST` inside the bash file of [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh). 

 - [ ] After initial installation, `DATA_HOST=$HOME_HOST"/dataset"` which should contain folder `/study_1_brain_flair`, folder `/study_1_stroke`, and CSV file of `study_data_list.csv` after initial cloning of this repository. Please open the `PUT .nii.gz FILES HERE.txt` files inside the two folders and the `study_data_list.csv` file for examples.
 - [ ] Remember, inside the folder `DATA_HOST`, there must be 2 folders (each containing T2 FLAIR brain MRI scans and their corresponding stroke lesions masks) and 1 CSV file which containing list of names T2 FLAIR brain MRI scans that need to be run in the inference. All data should be in *.nii.gz where the skull of head MRI have been removed (i.e., only brain). Again, please open the `PUT .nii.gz FILES HERE.txt` files inside the two folders and the `study_data_list.csv` file for examples.
 - [ ] If you want to change the names of folder containing T2-FLAIR brain MRI and their corresponding stroke lesion masks, you can do that by changing the variables `T2_FLAIR_BRAIN_DIR` and `STROKE_MASK_DIR`, respectively. The output folder's name can be change via variable  `OUTPUT_VOLUME_DIR`. The CSV file's name can be changed via variable `STUDY_DATA_LIST`.

# FOR DEV ONLY (Not recommended if you only need inference process without changing pre-processing etc.)

If you want to change how the inference process work (e.g., pre-processing, post-processing), you can use the bash file of [`predict-dev.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict-dev.sh) which will run [`./wmh-prediction/dev/inference-dev.py`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/dev/inference-dev.py). The bash file `predict-dev.sh` has similar variables like the [`predict.sh`](https://github.com/febrianrachmadi/wmh-prediction/blob/main/predict.sh) with several modifications.

# Create conda env on you machine instead of Docker containers

You can rename the current file by clicking the file name in the navigation bar or by clicking the **Rename** button in the file explorer.

 - [ ] NVIDIA GPUs driver and cuda
 - [ ] pip install tensorflow[and-cu da]==2.12.0 (tensorflow with GPU version 2.12 tested just fine)
 - [ ] pip install gdown
 - [ ] pip install pandas
 - [ ] pip install nibabel
 - [ ] pip install tabulate
 - [ ] pip install matplotlib
 - [ ] pip install scikit-learn
 - [ ] pip install tensorflow-addons

# Questions?

Please let me know should you have any questions via GitHub. I will make sure to check them regularly..
