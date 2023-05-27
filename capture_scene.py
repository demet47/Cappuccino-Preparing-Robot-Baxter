import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt

class Capture:
    def __init__(self):

# Load the model
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.counter = 0
        self.pipeline.start(self.config)

    def take_ss(self):
        flag = False
        image_name = 'image_' + str(self.counter) +'.png'
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                
                if(flag == False):
                   flag = True
                   continue
                
                cv2.imwrite("./image_captures/" + image_name, color_image)
                self.counter = self.counter + 1
                
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break
                    
                #if cv2.waitKey(1) & 0xFF == ord('s'):
                #    cv2.imwrite('./screenshots/image_' + str(counter) +'.png', color_image)
                #    counter = counter + 1
                #    print("Image saved!")

                break
        finally:
            #self.pipeline.stop()
            m = 3
        return image_name




#HOW SCREENCAPTURE IS TAKEN: an rgb camera view pops up on the screen when we run the code.
#you have to press s while your mouse is on the screen to save a capture.

#TODO: we will change this trigger event to our purpose