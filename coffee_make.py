import screenshot.capture_scene as ss
from roboflow import Roboflow
import models
import torch


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
            if((x <= self.restriction_right) and (x >= self.restriction_left) and (y >= self.restriction_upper) and (y <= self.restriction_lower)):
                return x,y
        return None, None

    def prepare(self, low_sugar):
        ss_name = self.screen.take_ss() #name of the screen shot .png file

        # below we give the image recognition model this image name and receive the coordinates for cup
        rf = Roboflow(api_key="o03639Rjl20zIjHrKB4v") #TODO: remove this api key and set it as environment variable
        project = rf.workspace("boazii-university").project("cup_place_finder")
        dataset = project.version(1).download("yolov5")
        model = project.version(dataset.version).model
        pred = model.predict("./screenshot/" + ss_name, confidence=70, overlap=30).json()
        x,_ = self.__get_location__(pred)
        if x == None:
            print("Error detecting the bounding boxes.")
            return
        #TODO: give the coordinates of cup to cnmp trained model and receive a trajectory
        model = models.CNP((3, 16), 256, 2, 0.01)
        state_dict = torch.load("./trained_models/cnp.pt")
        model.load_state_dict(state_dict["model_state_dict"])
        model.eval()
        data = torch.load("initial_data/trajectoriessalimdemet.pt") #TODO: orient to our case
        traj = data["salimdemet1_0.csv"]
        traj = traj.unsqueeze(0)
        mean, std = model(observation=traj[:, :], target=traj[:, :, [0]])
        mean = mean.detach()
        std = std.detach()

        model(observation=traj[:, [0, 1, 3000, 3001]],
            target=torch.tensor([
                [
                    [0.1, 0.3,0.4],
                    [0.5, 0.3, 0.4],
                    [0.7, 0.3, 0.4],
                ]
            ]))
        #TODO: save the trajectory in proper format as in the lab
        
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