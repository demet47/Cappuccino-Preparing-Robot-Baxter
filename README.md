
## Dictionary
- _raw trajectory data:_ the trajectory data collected via related script inside Baxter server. It originally contains 17 columns: 1 time, 8 left arm joints, 8 right arm joints. In some contexts, this can refer to a modified version, 2 columns added: x, y pixel location coordinates for the object we seek to reach as goal with the related trajectory.

## Files and Where They are User

- *main_code.py*:
    - this is the script we run at the uppermost layer for the cyclic orchestration of coffee make.
    - contains voice recognition
    - makes a call to coffe make module
- *execute_remote.py*:
    - it takes an argument when running from terminal. 
        - argument is the bash command we want the script to run
    - the execution of the command happens at the remote. This code internally connects to Baxter server and executes the given command in it's environment.
- *coffee_make.py*:
    - ___get_location___: takes prediction dictionary produced by image detection model and gives the x-y value predicted for the location of the coffee cup
    - ___produce_trajectory___: takes the x-y of the cup as input and gives produces the trajectory with these two input data.
    - prepare: does image capture, image detection via a trained model, simulates the coffee making process with predicting the trajectory for repositioning the cup, executing the rest of the hardcoded processes by connecting to the remote. 
- *capture_scene.py*:
    - for connecting to intelisense camera and taking a capture of the current setting, saving it to __./image_captures__ directory 
  *pytorch_models.py:*
    - different types of model classes for training in CNP.
  *ssh_send_with_sftp.py:*
    - takes a file path as argument and sends the path to Baxter server using ssh.
  *Task Parameterization and Generalization.ipynb:*
    - training code for CNMP model.
  *trajectory training raw data collection.md*:
    - documentation on how to collect raw trajectory data using Baxter server.
- *.env*:
    - contains API key for the image detection remote folder
    - contains SSH connection information to Baxter server
- *data_utils.py*:
    - file containing functions for raw trajectory  file manipulation and conversion to certain formats.
## Folders
- *train_images*:
    - images used to train the image recognition model on remote.
- *sound_files*:
    - where the baxter played speech .mp3 files are collected (both static and dynamic ones)
- *object detection*:
    - folder containing codes for image detection
- *carry_data*:
    - raw trajectory data for training of cup replacement. collected in lab. 
    - detailed explanation of how we collected can be found in file _trajectory training raw data collection.md_
    - grouped as training and validation, for different training and validation sizes into different folders.

- *code_inventory*:
    - codes we wrote during the project building process. We use code pieces from it from time to time.
    - *folders:*
        - *data_manipulation:*
            - _add\_xy\_column.py:_ adds the pixel location data we want to the raw trajectory data.
        - *image_detection_scripts:*
            - _detect\_objects\_torchvision:_ detects bounding boxes around objects and displays labels on the detected objects on image. Utilizes _Faster RCNN using ResNet-101_
            - _yolov5\_detection.py:_ uses pretrained yolov5 model from "model zoo" for detecting bounding boxes around objects. It displays the picture with detected bounding boxes at termination.
        - *liquid\_pour\_train\_data:* raw trajectory data for pouring liquid, starting from different locations to a cup at a certain position. The initial locations of cups for related trajectories are coded in file namings. 
        - *manually\_prepared\_trajectories:* contains trajectories that are reflected in the file namings. Were created to experiment with Pybullet. Can be imported inside Python code as numpy arrays.
        - *sample_train_images:* images for fine-tuning image detection custom model. These spesifically contain tabletop view of ciruclar edge top-viewed images.
        - *simulator:* files for working with Pybullet
            - _custom\_visualize\_in\_pybullet:_ running the file prompts user to enter a text input from following list: ["linear", "circular", "step", "sinus", "triangular", "polynomial", "impuls", "multi-step"]. Then it demonstrates from the implicit related trajectories according to selection.
            - _manipulator:_ Class contains functions to interact with Baxter object. 
            - _sample\_data\_formation:_ code for creating sample trajectories and saving them as .npy files.
            - _set\_environment:_ code for importing metadata from baxter_common folder and initiating a Pybullet screen that demonstrates Baxter. The second function inside is for importing objects other than Baxter into the screen.
                - *important:* the .urdf files are imported from a special folder. Detailed information of the objects to be imported can be found in this folder's README.
            - _visualize\_in\_pybullet:_ non-modified version of custom_visualize_in_pybullet.py file. Directly taken from CoLoRs Lab code inventory.
        - *table\_object\_pybullet:* content of this folder should be directly put to the directory mentioned in previous directory's README file. This is a table object.
- *cup_place_finderv2-1*: 
    - folder utilized when finetuning using Roboflow
- *image_captures*: 
    - folder where images snasphots collected from Intelisense are recorded during coffee making process.
- *left_arm*: 
    - we trained the CNMP model using CUDA. The computer we ran the program loading the trained model neither had CUDA nor GPU. Thus, we had to find a special way to load the model. 
    - this folder is a special folder with an appropriate format to load inside Python code as a model.
- *meta_trajectories*: 
    - trajectories that are hardcoded and utilized on midway.
    - files:
        - _left\_arm\_default\_data:_ the trajectory data that demonstrates the stationary positioning of left arm for a randomly selected time length
        - _right\_arm\_default\_data:_ the trajectory data that demonstrates the stationary positioning of right arm for a randomly selected time length
        - _train\_joints\_left_: first 20 and last 20 time steps for a random cup replacement trajectory for left arm. Is used to feed the prediction process as input. (comes from the gaussian prediction process)
        -  _train\_joints\_right_: first 20 and last 20 time steps for a random cup replacement trajectory for right arm. Is used to feed the prediction process as input. (comes from the gaussian prediction process)
- *object detection:* 
    - folder used for bounding box detection finetuning using Roboflow
- *right_arm:* 
    - right arm version of the _left\_arm_ folder
- *sound_files:* 
    - mp3 files used inside the coffee makinc main code. 
    - files are used when demonstrating voice directive and autonomous execution using voice communication. Some change when running the main code, some never change and are hardcoded. These are documented inside the main code.
- *train_images:* 
    - table-top view images of the cup located at special positions as encoded in the image name.
- *train_models:* 
    - contains trained models.
- *training_graphs:* 
    - folder containing images of the graphs to evaluate the success (precision of the predictions) of trained models.
- *trajectories:* 
    - the folder contains predicted trajectories. 
    - their format is such that it can be directly sent to the Baxter server and run there with special directives. - The files are formed implicitely inside main code according to given inputs.
- *colors-lab codes*
    - codes from the colors-lab repository mainly for CNP training and data manipulation-extraction.
    - contains below codes we use a lot:
        - *rec2pt.py*: takes input folder name and output file name as arguments. produces a torch file containing a list of multiple trajectories that existed inside the input folder with the extension .csv
    - *train.py*
        - script we use for training the model we use while coffee cup replacement
    - *config.yaml*
        - configuration file for training script
- *baxter_common*:
    - common files for baxter configuration. taken from lab repo. never modified.
    - it found usage in our case only during using simulator _Pybullet_


