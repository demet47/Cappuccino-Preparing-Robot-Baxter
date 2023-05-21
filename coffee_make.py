import screenshot.capture_scene as ss
from roboflow import Roboflow
import models
import torch
import subprocess
import numpy as np


class Coffee:
    def __init__(self):
        self.screen = ss.Capture()
        self.restriction_upper = 0
        self.restriction_lower = 0
        self.restriction_left = 0
        self.restriction_right = 0

    def __get_location__(self, prediction_dictionary):
        for box in prediction_dictionary['predictions']:
            x = box["x"]
            y = box["y"]
            return x,y

    def prepare(self, low_sugar):
        ss_name = self.screen.take_ss() #name of the screen shot .png file

        # below we give the image recognition model this image name and receive the coordinates for cup
        rf = Roboflow(api_key="o03639Rjl20zIjHrKB4v")
        project = rf.workspace("boazii-university").project("cup_place_finderv2")
        dataset = project.version(1).download("yolov5")
        model = project.version(dataset.version).model
        pred = model.predict("./screenshot/screenshots/" + ss_name, confidence=70, overlap=30).json()
        x,_ = self.__get_location__(pred)
        if x == None:
            print("Error detecting the bounding boxes.")
            return
        #TODO: give the coordinates of cup to cnmp trained model and receive a trajectory
        model = models.CNP((3, 16), 256, 2, 0.01)
        state_dict = torch.load("./colors-lab codes/train_scripts/save/deneme1/model.pt", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        data = torch.load("./carry_data/val/val.pt") #TODO: orient to our case
        traj = data["carry_10_3.csv"]
        traj = traj.unsqueeze(0)
        mean, std = model(observation=traj[:, :], target=traj[:, :, [0,1,2]])
        mean = mean.detach()
        std = std.detach()

        predicted_trajectory = model(observation=traj[:, [0, 1, 3000, 3001]],
            target=traj[0,:,0:3].unsqueeze(0))
        values = predicted_trajectory[0].squeeze(0).detach().numpy()
        headers = np.array(["time","left_s0","left_s1","left_e0","left_e1","left_w0","left_w1","left_w2","left_gripper","right_s0","right_s1","right_e0","right_e1","right_w0","right_w1","right_w2","right_gripper"])
        headers = headers.reshape(1,17)

        time_len = values.shape[0]
        times = np.arange(0.45, 0.45 + time_len * 0.01, 0.01).reshape(-1, 1)
        values = np.hstack((times,values))
        record = np.vstack((headers,values))

        np.savetxt('output.csv', record, delimiter=',', fmt='%s')
        #TODO: save the trajectory in proper format as in the lab
        subprocess.call(["python", "other_code.py"]) #example code to run bash command
        #TODO: make the robot execute this saved trajectories
        if(low_sugar):
            #TODO: make the robot execute the low sugar trjectory file
            a = ""
        else:
            #TODO: make the robot execute the high sugar trajectory file
            b = ""


# BELOW IS A SAMPLE CODE TO CALL A PYTHON EXECUTABLE FILE AND RUN IT
'''
if any(key in text.lower() for key in keywords):
            # if keyword in text.lower():
                # do something here, like play a sound or send a notification
                # os.system("aplay /usr/share/sounds/alsa/Front_Center.wav")
                with open("other_code.py", "r") as f:
                    exec(f.read())
                print("Detected keyword: " + keyword)
'''