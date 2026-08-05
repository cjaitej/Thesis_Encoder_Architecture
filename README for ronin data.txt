RoNIN: Robust Neural Inertial Navigation
Project website: https://ronin.cs.sfu.ca/

--------------------
GENERAL INFORMATION
--------------------

1. Title of Dataset: RoNIN: Robust Neural Inertial Navigation

2. Author Information

	A. Principal Investigator Contact Information
		Name: Yasutaka Furukawa
		Institution: Simon Fraser University 
		Email: furukawa@sfu.ca

	B. Associate or Co-investigator Contact Information
		Name: Sachini Herath
		Institution: Simon Fraser University
		Email: sherath@sfu.ca

	C. Associate or Co-investigator Contact Information
		Name: Hang Yan
		Institution: Washington University 


3. Information about funding sources that supported this research: 

- National Science Foundation (NSF): IIS 1618685
- Natural Sciences and Engineering Research Council of Canada (NSERC): Discovery Grants
- Natural Sciences and Engineering Research Council of Canada (NSERC): Discovery Accelerator Supplements
- Department of National Defence (DND) and Natural Sciences and Engineering Research Council of Canada (NSERC): Discovery Grant Supplements

------------------------------------------
SHARING/ACCESS INFORMATION - DATA & MODEL
------------------------------------------

1. Licenses/restrictions placed on the data and model: 

The RoNIN Data and Model are made available under a custom license for non-commercial and non-commercial scientific research purposes only. Please see the appended LICENSE.txt file for full license details and restrictions. 

If you report results with this dataset, you agree to cite the following paper: 

Sachini Herath, Hang Yan, and Yasutaka Furukawa. RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods. 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 3146-3152. https://doi.org/10.1109/ICRA40945.2020.9196860


2. Links to publications that cite or use the data: 

Sachini Herath, Hang Yan, and Yasutaka Furukawa. RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, & New Methods. 2020 IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 3146-3152. https://doi.org/10.1109/ICRA40945.2020.9196860

Hang Yan, Sachini Herath, and Yasutaka Furukawa. RoNIN: Robust Neural Inertial Navigation in the Wild: Benchmark, Evaluations, and New Methods. arXiv preprint arXiv:1905.12853.

3. Links/relationships to ancillary data sets or software packages: 

RoNIN Source Code: https://github.com/Sachini/ronin

4. Was data derived from another source? No

5. Recommended citation for this dataset: 

Herath S, Furukawa Y, Yan H. (2022). RoNIN: Robust Neural Inertial Navigation. Federated Research Data Repository. https://doi.org/10.20383/102.0543

If you use this dataset, please also cite the associated publication, referenced above.


---------------------
DATA & FILE OVERVIEW
---------------------

1. File List

Data
   
   Each data sequence is in a folder with name format "<subject_identifier>_<sequence_identifier>". A folder contains two files: 
       - "data.hdf5" - contains raw and time synchronized data from two smartphones. (See below for data description.)
       - "info.json" - contains meta information such as sensor calibration, time syncronization. (See below for meta data description.)

   A. Filename: train_dataset_1.zip   
      Short description: Training and validation data sequences       

   B. Filename: train_dataset_2.zip   
      Short description: Training and validation data sequences

   C. Filename: seen_subjects_test_set.zip     
      Short description: Testing data sequences from subjects that appear in training dataset

   D. Filename: unseen_subjects_test_set.zip      
      Short description: Testing data sequences from subjects that does not appear in training dataset


Pretrained_Models 

   Each folder contains the best checkpoint we obtained for each task and backbone architecture. The network configurations are specified as default configuration in the source code and optionally in the config.json file included with the checkpoint.

   F. Filename: ronin_resnet.zip   
      Short description: RoNIN ResNet model for estimating relative position of the subject from IMU data.      

   G. Filename: ronin_lstm.zip  
      Short description: RoNIN LSTM model for estimating relative position of the subject from IMU data      

   H. Filename: ronin_tcn.zip     
      Short description: RoNIN TCN model for estimating relative position of the subject from IMU data

   I. Filename: ronin_body_heading.zip     
      Short description: RoNIN LSTM model for estimating relative body heading of the subject from IMU data


3. Additional related research objects that are not included in the current data package: 

Model code is shared on Github (https://github.com/Sachini/ronin). 

Due to security concerns, we are unable to publish ~ 50% of our dataset. As such, the pre-trained models published here cannot be fully reproduced using the model code in Github and the data published here (the models were trained on the larger dataset, which includes private data). 


---------------------------
METHODOLOGICAL INFORMATION
---------------------------


1. Description of methods used for collection/generation of data: 

Please refer to our supplementary document (https://arxiv.org/pdf/1905.12853.pdf, pages 10-11) for the details of data collection. The Android applications developed for data collection can be found on Github (https://github.com/Sachini/ronin).


2. Methods for processing the data: 

Please refer to our supplementary document (https://arxiv.org/pdf/1905.12853.pdf, pages 10-11) for the details of data processing steps. The source code for processing data from a single device (for inference only) can be found at https://github.com/Sachini/ronin/tree/master/source/preprocessing.


3. Instrument- or software-specific information needed to interpret the data: 
Any compatible libraries maybe use to read hdf5 and json files. We used h5py python library and the default json library for the task.


--------------------------
DATA-SPECIFIC INFORMATION 
--------------------------

Data collection for this study was approved by the Simon Fraser University Office of Research Ethics, and study participants provided informed consent. 


RoNIN Dataset
-------------
The dataset contains 327 sequences in total in the format
	sequence_name
		|--- data.hdf5
		|--- info.json

Unfortunately, due to security concerns we were unable to publish ~ 50% of our dataset.


HDF5 data format
-----------------
data.hdf5
     |---raw
     |    |---tango
     |          |---gyro, gyro_uncalib, acce, magnet, game_rv, gravity, linacce, tango_pose, tango_adf_pose, rv, pressure, (optional) [step,wifi, gps, magnetic_rv, magnet_uncalib]
     |    |--- imu
     |          |---gyro, gyro_uncalib, acce, magnet, game_rv, gravity, linacce, rv, pressure, (optional) [step, wifi, gps, magnetic_rv, magnet_uncalib]
     |--synced
     |    |--- time, gyro, gyro_uncalib, acce, magnet, game_rv, rv, gravity, linacce
     |---pose
     |    |---tango_pos, tango_ori, (optional) ekf_ori



HDF5 data description
---------------------
Please refer to official documentation at Android developers (https://developer.android.com/) and Google Tango (Archived - https://web.archive.org/web/20170714191228/https://developers.google.com/tango/apis/unity/) for data specification from each API.

Data sources
	 - gyro		- Android Sensor.TYPE_GYROSCOPE
	 			- A device's rate of rotation in rad/s around each of the three physical axes (x, y, and z)
	 - gyro_uncalib	- Android Sensor.TYPE_GYROSCOPE_UNCALIBRATED
	 			- A device's rate of rotation in rad/s around each of the three physical axes (x, y, and z) without drift compenstation. 
	 - acce		- Android Sensor.TYPE_ACCELEROMETER
	 			- The acceleration force in m/s2 that is applied to a device on all three physical axes (x, y, and z), including the force of gravity.
	 - linacce		- Android Sensor.TYPE_LINEAR_ACCELERATION
	 			- The acceleration force in m/s2 that is applied to a device on all three physical axes (x, y, and z), excluding the force of gravity.
	 - gravity		- Android Sensor.TYPE_GRAVITY
	 			- The force of gravity in m/s2 that is applied to a device on all three physical axes (x, y, z).
	 - magnet		- Android Sensor.TYPE_MAGNETIC_FIELD
	 			- The ambient geomagnetic field for all three physical axes (x, y, z) in μT.
	 - magnet_uncalib 	- Android Sensor.TYPE_MAGNETIC_FIELD_UNCALIBRATED
	 			- The ambient geomagnetic field for all three physical axes (x, y, z) in μT without hard iron calibration.
	 - rv		 	- Android Sensor.TYPE_ROTATION_VECTOR
	 			- The orientation of a device by providing the device's rotation vector as a quaternion (x,y,z,w) calculated using accelerometer, gyroscope and magnetometer.
	 - game_rv		- Android Sensor.TYPE_GAME_ROTATION_VECTOR
	 			- The orientation of a device by providing the device's rotation vector as a quaternion (x,y,z,w) calculated using accelerometer and gyroscope.
	 - magnetic_rv		- Android Sensor.TYPE_GEOMAGNETIC_ROTATION_VECTOR
	 			- The orientation of a device by providing the device's rotation vector as a quaternion (x,y,z,w) calculated using accelerometer and magnetometer.
	 - step		- Android Sensor.TYPE_STEP_COUNTER
	 			- the number of steps taken by the user while activated
	 - pressure		- Android Sensor.TYPE_PRESSURE
	 			- Atmospheric pressure in hPa (millibar)
	 - gps			- Android LocationManager.GPS_PROVIDER
	 			- Location data as (latitude, longitude, altitude, accuracy, bearing, speed)
	 - tango_pose		- Pose from Visual SLAM of Tango device (format: time, position (x,y,z), orientation (x,y,z,w))
	 - tango_adf__pose	- Pose from Visual SLAM with area learning of Tango device
	 - wifi		- wifi footprints scanned every 3 seconds. stored in 2 parts
		|-- "wifi_values" - contains (scan_number, last_timestep, level) 
		|--"wifi_address" - dataset of type string. contains BSSID of corresponding records in wifi_values

"raw" group contains data as reported by APIs in format 
	- system_timestamp (nanosecond), API output
"synced" group contains time synchronized data from IMU device sampled at 200 Hz
    - time : System time of IMU device in seconds
"pose" group store all pose information (timestamp for data is "synced/time")
    - tango_pos, tango_ori - tango_pose and tango_adf_pose combined for more accurate pose estimation
    - ekf_ori - orientation of IMU device calculated using sensor fusion. (Should not be used during testing)


JSON data description
---------------------
  	Stores meta information
	 - length	: sequence length in seconds
	 - date  	: collected date mm/dd/yy
	 - device	: device_identifier

	 - align_tango_to_body : approx. alignment between Tango device coordinate frame and body 
 	 - start_frame		   : dataframe which marks end of pre-calibration and synchronization. (use data from this point forward)

	Sensor calibration of IMU device. (Please refer supplementary material for details)	
	 - imu_init_gyro_bias
	 - imu_end_gyro_bias 
	 - imu_acce_bias
	 - imu_acce_scale

	Time syncronization between the system time of two devices 
	t_imu = t_tango - tango_reference_time  + imu_reference_time + imu_time_offset
	 - imu_reference_time	: approx. time alignments (nanoseconds) 
	 - tango_reference_time : approx. time alignments (nanoseconds) 
	 - imu_time_offset: 	: fine tuned time alignments (seconds)
	
	Alignment between Tango reference coordinate frame and game rotation vector of IMU device
	 - start_calibration 
	 - end_calibration
	
	Orientation drift calculated using end_calibration 
	 - gyro_integration_error
	 - grv_ori_error		 
	 - ekf_ori_error

---------------------------
MODEL-SPECIFIC INFORMATION 
---------------------------
1. Link to code repository: https://github.com/Sachini/ronin

2. Brief description of the Model:

The models are Python Pytorch checkpoints used in our final evaluation. The models maybe used as pretraining weights for training or to evaluate on RoNIN or a similar dataset. Please refer to the README file in Github for commands for launching train / test code.

3. Required dependencies: 

Please refer to the requirements.txt file in Github for the complete list of python dependencies we have used.
