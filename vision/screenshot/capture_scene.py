import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

counter = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('RealSense', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('./screenshots/image_' + str(counter) +'.png', color_image)
            counter = counter + 1
            print("Image saved!")
            
finally:
    pipeline.stop()
    cv2.destroyAllWindows()



#HOW SCREENCAPTURE IS TAKEN: an rgb camera view pops up on the screen when we run the code.
#you have to press s while your mouse is on the screen to save a capture.

#TODO: we will change this trigger event to our purpose