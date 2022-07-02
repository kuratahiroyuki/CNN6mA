# CNN6mA
This package is used for 6mA site prediction

# Features
・CNN6mA predicts 6mA site by sequence information.    

# Environment
    Python   : 3.8.0
    Anaconda : 4.9.2
※We recommend creating virtual environments by using anaconda.

# Processing
 This CLI system is used for three processing as follows.  
 ・Training of CNN6mA model for 6mA site prediction.  
 ・6mA site prediction by the trained model.  

# preparation and installation
## 0. Preparation of a virtual environment (not necessary)
0-1. Creating a virtual environment  
    `$conda create -n [virtual environment name] python==3.8.0`
    ex)  
    `$conda create -n cnn6mA python==3.8.0`
      
0-2. Activating the virtual environment  
    `$ conda activate [virtual environment name]`
    ex)  
    `$ conda activate cnn6mA`
    
0-3. Installing pytorch
    Pytorch with the version which adjusts your Cuda version need to be installed.  
    Refer to the following Pytorch sites for the corresponding of Cuda and Pytorch versions.  
    For example, if you use Cuda with version 11.3, Pytorch is installed by the following.  
    `$pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113`
    If you use cpu, Pytorch is installed by the following.  
    `$pip install torch`
    
## 1. Installing the CNN6mA package and pytorch
CNN6mA is installed by executing the following command in the directory where the package is located.
`$pip install ./CNN6mA/dist/CNN6mA-0.0.1.tar.tar.gz`

## 2. Training of CNN6mA model for 6mA site prediction
CNN6mA model for 6mA prediction can be trained by following command.  
`$cnn6mA train -t [Training data file path (csv format)] -v [Training data file path (csv format)] -o [output dir path]`

ex)  
`$cnn6mA train -t ./CNN6mA/sample_data/sample_train_data.csv -v ./CNN6mA/sample_data/sample_val_data.csv -o ./CNN6mA/results`

Note that csv files need to contein the following contents (Check the sample data at /CNN6mA/sample_data)  
First column (seq):DNE sequence with a length of 41 bp
Second column (labels):label (1: 6mA, 0: non-6mA)  

other options)
|option|explanation|necessary or not|default value|
|:----:|:----:|:----:|:----:|
|-t (--training_file)|Path of training data file (.csv)|necessary|-|
|-v (--validation_file)|Path of validation data file (.csv)|necessary|-|
|-o (--out_dir)|Directory to output the trained model|necessary|-|
|-t_batch (--training_batch_size)|Training batch size|not necessary|32|
|-v_batch (--validation_batch_size)|Validation batch size|not necessary|32|
|-lr (--learning_rate)|Learning rate|not necessary|0.0001|
|-max_epoch (--max_epoch_num)|Maximum epoch number|not necessary|10000|
|-stop_epoch (--early_stopping_epoch_num)|Epoch number for early stopping|not necessary|20|
|-thr (--threshold)|Threshold to determined whether interact or not|not necessary|0.5|
|-seq_len (--sequence_length)|Sequence length|not necessary|41|
|-tp (--target_pos)|Modification site position|not necessary|9000|
|-device (--device)|Device to be used. If you use cpu, "cpu" must be specified|not necessary|cuda:0|

(Results)  
model files will be output to the specified directory.  
Filename: deep_model
|Filename|contents|
|:----:|:----:|
|deep_model|CNN6mA model file|

## 3.6mA site prediction
6mA site prediction is executed by following command.  
`$cnn6mA predict -i [data file path (csv format)] -o [output dir path] -d [deep learning model file path]`

ex)  
`$cnn6mA predict -i ./CNN6mA/sample_data/sample_test_data.csv -o ./CNN6mA/results -d ./CNN6mA/data_model/6mA_A.thaliana/deep_model`

other options)
|option|explanation|necessary or not|default value|
|:----:|:----:|:----:|:----:|
|-i (--import_file)|Path of data file (.csv)|necessary|-|
|-o (--out_dir)|Directory to output results|necessary|-|
|-d (--deep_model_file)|Path of a trained attention-phv model|necessary|-|
|-vec (--vec_index)|Flag whether contribution score vectors output|not necessary|False|
|-thr (--threshold)|Threshold to determined whether interact or not|not necessary|0.5|
|-batch (--batch_size)|Batch size|not necessary|32|
|-seq_len (--sequence_length)|Sequence length|not necessary|41|
|-tp (--target_pos)|Modification site position|not necessary|9000|
|-device (--device)|Device to be used. If you use cpu, "cpu" must be specified|not necessary|cuda:0|

Note that csv files need to contein the following contents (Check the sample data at /CNN6mA/sample_data)  
First column (seq):DNE sequence with a length of 41 bp

(Results)  
CSV files will be output to the specified directory.  
Filename: prediction_results.csv, score_vec_list.csv (option)

|Filename|contents|
|:----:|:----:|
|prediction_results.csv|Predictive scores|
|score_vec_list.csv (option)|Contribution score vectors|

#  Other contents
We provided sample data and CNN6mA model as well as CLI system.
Note that sample data is not the benchmark datasets and this is only present as an example.

              














