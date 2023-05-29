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
- *.env*:
    - contains API key for the image detection remote folder
    - contains SSH connection information to Baxter server
## Folders
- *train_images*:
    - images used to train the image recognition model on remote.
- *sound_files*:
    - where the baxter played speech .mp3 files are collected (both static and dynamic ones)
- *object detection*:
    - folder containing codes for image detection
- *carry_data*:
    - trajectory data for training of cup replacement. collected in lab.
- *gibberish_code_might_be_useful*:
    - codes we wrote during the project building process. We use code pieces from it from time to time.
- *cup_place_finderv2-1*:
- *colors-lab codes*
    - codes from the colors-lab repository mainly for CNP training and data manipulation-extraction.
    - contains below codes we use a lot:
        - *rec2pt.py*: takes input folder name and output file name as arguments. produces a torch file containing a list of multiple trajectories that existed inside the input folder with the extension .csv
    - *train.py*
        - script we use for training the model we use while coffee cup replacement
    - *config.yaml*
        - configuration file for training script
- *baxter_common*:
    - common files for baxter configuration. taken from lab repo


