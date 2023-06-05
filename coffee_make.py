import capture_scene as ss
from roboflow import Roboflow
import models
import torch
import subprocess
import numpy as np
import time
import os
from dotenv import load_dotenv
import pygame
from gtts import gTTS
import tensorflow as tf
import tensorflow_probability as tfp
from keras.models import load_model
import keras.losses
import data_format

class Coffee:
    def __init__(self):
        self.screen = ss.Capture()
        self.index = 0
        self.traj_time_len = 4000 #time length the model is going to make prediction of
        self.n_max = 1



    def __get_location__(self, prediction_dictionary):
        for box in prediction_dictionary['predictions']:
            x = box["x"]
            y = box["y"]
            return x,y

    def __text_to_speech__(self,text, output_file):
        # Initialize the pyttsx3 engine
        tts = gTTS(
            text=text,
            lang="en"
        )
        tts.save(output_file)


    def __display_sound__(self,response_recording):
        pygame.mixer.music.load(response_recording)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)

    
    def custom_loss(self, y_true, y_pred):
        mean, log_sigma = tf.split(y_pred, 2, axis=-1)
        y_target, temp =tf.split(y_true,2,axis=-1)
        sigma = tf.nn.softplus(log_sigma)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=sigma)
        loss = -tf.reduce_mean(dist.log_prob(y_target))
        return loss


    def __produce_trajectory__(self, x, y): #TODO: entegrate here x and y
        keras.losses.custom_loss = self.custom_loss
        model = 0 #gibberish value just for referencing
        if x > 0.5: #TODO: not sure
            model = load_model('./trained_models/task_parametrization_left.h5', custom_objects={ 'tf':tf })
        else:
            model = load_model('./trained_models/task_parametrization_right.h5', custom_objects={ 'tf':tf })
        
        prediction = np.zeros((8,self.traj_time_len))
        observation = np.zeros((1,1,11))
        observation_flag = np.zeros((1,1,1))
        target = np.zeros((1,1,3))
        ob_p=x
        w_p=y

        if x > 0.5:
            #for time in range(20):
            observation[0,0] = [0,ob_p,w_p,
                                    data_format.observation_left[0,0,0],
                                    data_format.observation_left[0,1,0],
                                    data_format.observation_left[0,2,0],
                                    data_format.observation_left[0,3,0],
                                    data_format.observation_left[0,4,0],
                                    data_format.observation_left[0,5,0],
                                    data_format.observation_left[0,6,0],
                                    data_format.observation_left[0,7,0]]
            observation_flag[0,0,0] = 1.
                
        else:
            #for time in range(self.traj_time_len):
            observation[0,0] = [0,ob_p,w_p,
                                    data_format.observation_right[0,0,0],
                                    data_format.observation_right[0,1,0],
                                    data_format.observation_right[0,2,0],
                                    data_format.observation_right[0,3,0],
                                    data_format.observation_right[0,4,0],
                                    data_format.observation_right[0,5,0],
                                    data_format.observation_right[0,6,0],
                                    data_format.observation_right[0,7,0]]
            observation_flag[0,0,0] = 1.

        for i in range(self.traj_time_len):
            target[0,0] = [i/self.traj_time_len,ob_p,w_p]
            p = model.predict([observation,observation_flag,target])[0][0]
            prediction[:,i] = p[:8] #prediction.shape will be (8, self.traj_time_len) at the end
        
        
        return prediction


        '''
        FOR PYTORCH TRAINED MODEL:
        state_dict = torch.load("./colors-lab codes/train_scripts/save/deneme1/model.pt", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        data = torch.load("./carry_data/val/val.pt") #TODO: orient to our case
        traj = data["carry_10_3.csv"]
        traj = traj.unsqueeze(0)
        mean, std = model(observation=traj[:, :], target=traj[:, :, [0,1,2]])
        mean = mean.detach()
        std = std.detach()

        query = torch.cat((traj[0][:, [0]], torch.asarray([x,y]*traj.shape[1]).reshape(traj.shape[1],2)), dim=1)
        #TODO: beware, here since there are two arms, for better prediction there should be a check for which direction it is classified and choose one accordingly
        #by default, we chose carry_data_3.csv
        traj_time = traj.shape[1]
        predicted_trajectory = model(observation=traj[:, [i for i in range(0,20)]+[i for i in range(traj_time - 20, traj_time-1)], :],
            target=query.unsqueeze(0))
        values = predicted_trajectory[0].squeeze(0).detach().numpy()
        headers = np.array(["time","left_s0","left_s1","left_e0","left_e1","left_w0","left_w1","left_w2","left_gripper","right_s0","right_s1","right_e0","right_e1","right_w0","right_w1","right_w2","right_gripper"])
        headers = headers.reshape(1,17)

        time_len = values.shape[0]
        times = np.arange(0.45, 0.45 + time_len * 0.01, 0.01).reshape(-1, 1)
        values = np.hstack((times,values))
        record = np.vstack((headers,values))

        file_name = "output_" + str(self.index) + ".csv"
        self.index = self.index + 1
        np.savetxt(file_name, record, delimiter=',', fmt='%s')
        return file_name
        '''
        

        

    def __execute_remote__(self, display_file):
        command = "cd alper_workspace; source activate_env.sh; rosrun baxter_examples joint_trajectory_file_playback.py -f ../trajectories/" + display_file
        subprocess.call(["python", "./execute_remote.py", command], shell=True)


    def prepare(self, low_sugar):
        try:
            load_dotenv()

            ss_name = self.screen.take_ss() #name of the screen shot .png file

            # below we give the image recognition model this image name and receive the coordinates for cup
            rf = Roboflow(api_key=os.getenv("API_KEY"))
            project = rf.workspace(os.getenv("API_WORKSPACE")).project(os.getenv("API_PROJECT"))
            dataset = project.version(1).download("yolov5")
            model = project.version(dataset.version).model
            pred = model.predict("./image_captures/" + ss_name, confidence=70, overlap=30).json()
            x,y = self.__get_location__(pred)
            if x == None:
                print("Error detecting the bounding boxes.")
                return 1
            
            prediction = self.__produce_trajectory__(x,y)
            output_file_name = data_format.trajectory_list_to_csv(prediction.T) #TODO:self.__produce_trajectory__(x,y)

            subprocess.call(["python", "./ssh_send_with_sftp.py", output_file_name], shell=True)
            if(low_sugar):
                self.__execute_remote__("put_nescafe.csv")
                self.__execute_remote__("put_hot_water.csv")
                self.__execute_remote__("request_mixer.csv")
                text = "Milk mixer please."
                outfile = "./sound_files/mixer_request_from_baxter.mp3" #TODO:erase
                self.__text_to_speech__(text, outfile) #TODO:erase
                self.__display_sound__(outfile)
                time.sleep(5) #TIME IT TAKES FOR US TO GIVE MIZER TO BAXTER
                self.__display_sound__("mixer.csv")
                self.__display_sound__("put_milk.csv")
            else:
                self.__execute_remote__("put_nescafe.csv")
                self.__execute_remote__("put_hot_water.csv")
                self.__execute_remote__("put_sugar.csv")
                self.__execute_remote__("request_mixer.csv")
                text = "Milk mixer please."
                outfile = "./sound_files/mixer_request_from_baxter.mp3" #TODO:erase
                self.__text_to_speech__(text, outfile) #TODO:erase
                self.__display_sound__(outfile)
                time.sleep(5) #TIME IT TAKES FOR US TO GIVE MIZER TO BAXTER
                self.__execute_remote__("put_milk.csv")
            return 0
        except:
            return 2

a = Coffee()
a.__produce_trajectory__(0.6, 0.6)