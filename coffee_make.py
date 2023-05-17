import screenshot.capture_scene as ss




class Coffee:
    def __init__(self):
        self.screen = ss.Capture()

    def prepare(self, low_sugar):
        ss_name = self.screen.take_ss() #name of the screen shot .png file
        #TODO: give the image recognition model this image name and receive the coordinates for cup
        #TODO: give the coordinates of cup to cnmp trained model and receive a trajectory
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