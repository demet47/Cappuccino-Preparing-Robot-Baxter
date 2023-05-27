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

class Coffee:
    def __init__(self):
        #self.screen = ss.Capture()
        self.index = 0

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
    

    def __produce_trajectory__(self, x, y): #TODO: entegrate here x and y
        model = models.CNP((3, 16), 256, 2, 0.01)
        state_dict = torch.load("./colors-lab codes/save/deneme1/model.pt", map_location=torch.device('cpu'))
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

        file_name = "./trajectories/output_" + str(self.index) + ".csv"
        self.index = self.index + 1
        np.savetxt(file_name, record, delimiter=',', fmt='%s')
        return file_name


    def prepare(self, low_sugar):
        
        load_dotenv()
        '''
        ss_name = "image_0.png" #self.screen.take_ss() #name of the screen shot .png file TODO

        # below we give the image recognition model this image name and receive the coordinates for cup
        rf = Roboflow(api_key=os.getenv("API_KEY"))
        project = rf.workspace(os.getenv("API_WORKSPACE")).project(os.getenv("API_PROJECT"))
        dataset = project.version(1).download("yolov5")
        model = project.version(dataset.version).model
        pred = model.predict("./image_captures/" + ss_name, confidence=70, overlap=30).json()
        x,y = self.__get_location__(pred)
        if x == None:
            print("Error detecting the bounding boxes.")
            return
        #TODO: give the coordinates of cup to cnmp trained model and receive a trajectory
        '''
        
        
        
        output_file_name = self.__produce_trajectory__(1,2) #self.__produce_trajectory__(x,y)

        subprocess.call(["python", "./ssh_send_with_sftp.py", output_file_name], shell=True)
        if(low_sugar):
            command = "cd alper_workspace; source activate_env.sh; rosrun baxter_examples joint_trajectory_file_playback.py -f ../trajectories/low_sugar_part_1.csv"
            subprocess.call(["python", "./execute_remote.py", command], shell=True)
            text = "Milk mixer please."
            outfile = "./sound_files/mixer_request_from_baxter.mp3" #TODO:erase
            self.__text_to_speech__(text, outfile) #TODO:erase
            self.__display_sound__(outfile)
            time.sleep(4)
            command = "cd alper_workspace; source activate_env.sh; rosrun baxter_examples joint_trajectory_file_playback.py -f ../trajectories/low_sugar_part_2.csv"            
            subprocess.call(["python", "./execute_remote.py", command], shell=True)
        else:
            command = "cd alper_workspace; source activate_env.sh; rosrun baxter_examples joint_trajectory_file_playback.py -f ../trajectories/high_sugar_part_1.csv"
            subprocess.call(["python", "./execute_remote.py", command], shell=True)
            text = "Milk mixer please."
            outfile = "./sound_files/mixer_request_from_baxter.mp3" #TODO:erase
            self.__text_to_speech__(text, outfile) #TODO:erase
            self.__display_sound__(outfile)
            time.sleep(4)
            command = "cd alper_workspace; source activate_env.sh; rosrun baxter_examples joint_trajectory_file_playback.py -f ../trajectories/high_sugar_part_2.csv"            
            subprocess.call(["python", "./execute_remote.py", command], shell=True)

a = Coffee()
a.prepare(Tr)