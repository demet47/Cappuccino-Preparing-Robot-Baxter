import speech_recognition as sr
from contextlib import suppress
import coffee_make as coffee
from gtts import gTTS
import pygame
import time
import subprocess

state_counter = 0
bool_name = False
#initiate coffee maker
coffee_maker = coffee.Coffee()
name = ""

'''
VOICE FILES HARDCODED:
- ./sound_files/baxter_greeting.mp3
- ./sound_files/baxter_preparation_initalization_no_sugar.mp3
- ./sound_files/baxter_preparation_initalization_sugar.mp3
- ./sound_files/baxter_clarification_sugar.mp3
- ./sound_files/non-comprehending_baxter.mp3"
- ./sound_files/baxter_api_error.mp
- ./sound_files/mixer_request_from_baxter.mp3
- ./sound_files/baxter_farewell.mp3



TRAJECTORY FILES HARDCODED:
- ../trajectories/baxter_greet.csv
- ../trajectories/baxter_farewell.csv
- ../trajectories/put_nescafe.csv #TODO:ADD
- ../trajectories/put_hot_water.csv #TODO:ADD
- ../trajectories/put_sugar.csv #TODO:ADD
- ../trajectories/request_mixer.csv #TODO:ADD
- ../trajectories/mixer.csv #TODO:ADD
- ../trajectories/put_milk.csv #TODO:ADD


VOICE FILES NON-HARDCODED:
- ./sound_files/baxter_sugar_request.mp3
- ./sound_files/baxter_serve.mp3



TRAJECTORY FILES NON-HARDCODED:
- ./trajectories/output_x.csv

'''



def __text_to_speech__(text, output_file):
    # Initialize the pyttsx3 engine
    tts = gTTS(
        text=text,
        lang="en"
    )
    tts.save(output_file)



def __display_sound__(response_recording):
    pygame.mixer.music.load(response_recording)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)


with suppress(Exception):

    # set up the recognizer
    r = sr.Recognizer()

    # set up the microphone
    mic = sr.Microphone()

    #init pygame for mp3 playback
    pygame.mixer.init()

    # adjust the microphone sensitivity
    with mic as source:
        r.adjust_for_ambient_noise(source)


    # start listening
    print("Say something!")
    while True:

        with mic as source:
            audio = r.listen(source)

        # use Google speech recognition to transcribe the audio
        try:
            if bool_name:
                text = r.recognize_google(audio, language="tr-TR") 
            else:
                text = r.recognize_google(audio)

            print("You said: " + text)
            
            # check if the keyword is in the recognized text
            lower_case = text.lower()
            if((state_counter == 0) & (lower_case.__contains__("hello"))):
                baxter_response = "Hi there, so nice to have you here! Could you give me the honour of acquiring your name?" #TODO:erase
                response_recording = "./sound_files/baxter_greeting.mp3" #TODO: erase
                __text_to_speech__(baxter_response, response_recording) #TODO:erase
                __display_sound__(response_recording)
                command = "rosrun baxter_examples joint_trajectory_file_playback.py -f ./trajectories/baxter_greet.csv"
                call_wait = subprocess.Popen(["python", "./execute_remote.py", command], shell=True)
                call_wait.wait()
                state_counter = state_counter + 1
                print("In state: ", state_counter - 1)
                bool_name = True

            elif(state_counter == 1):
                name = text
                baxter_response = f"Welcome to the best cappuccino stand in the world {name}. I need to learn your sugar preference before I get started. Would you like sugar in your coffee?"
                response_recording = "./sound_files/baxter_sugar_request.mp3"
                __text_to_speech__(baxter_response, response_recording)
                __display_sound__(response_recording)
                state_counter = state_counter + 1
                print("In state: ", state_counter - 1)
                bool_name = False
            elif((state_counter == 2)):
                if(text.__contains__("no")):
                    baxter_response = "What a healthy choice congratulations! It will be ready in a minute." #TODO:erase
                    response_recording = "./sound_files/baxter_preparation_initalization_no_sugar.mp3" #TODO:erase
                    __text_to_speech__(baxter_response, response_recording) #TODO:erase
                    __display_sound__(response_recording)
                    return_flag = coffee_maker.prepare(low_sugar = True)
                    if(return_flag == 0):
                        state_counter = state_counter + 1
                        print("In state: ", state_counter - 1)
                    else:
                        baxter_response = "Woops! Something went wrong. Can you repeat your sugar preference?" 
                        response_recording = "./sound_files/baxter_prepare_error_recovery.mp3"
                        __text_to_speech__(baxter_response, response_recording)
                        __display_sound__(response_recording)
                    #TODO: Low sugar preparation
                elif(text.__contains__("yes")):
                    baxter_response = "Please beware, sugar is not that healthy a choice. It will be ready in a minute." #TODO:erase
                    response_recording = "./sound_files/baxter_preparation_initalization_sugar.mp3" #TODO:erase
                    __text_to_speech__(baxter_response, response_recording) #TODO:erase
                    __display_sound__(response_recording)
                    return_flag = coffee_maker.prepare(low_sugar = False)
                    if(return_flag == 0):
                        state_counter = state_counter + 1
                        print("In state: ", state_counter - 1)
                    else:
                        baxter_response = "Woops! Something went wrong. Can you repeat your sugar preference?" 
                        response_recording = "./sound_files/baxter_prepare_error_recovery.mp3"
                        __text_to_speech__(baxter_response, response_recording) 
                        __display_sound__(response_recording)
                else:
                    baxter_response = "I couldn't undestand. Could you please say merely yes or no?" #TODO:erase
                    response_recording = "./sound_files/baxter_clarification_sugar.mp3" #TODO:erase
                    __text_to_speech__(baxter_response, response_recording) #TODO:erase
                    __display_sound__(response_recording)
                    print("In state: ", state_counter)
            elif(state_counter == 3):
                baxter_response = f"Here you go {name}."
                response_recording = "./sound_files/baxter_serve.mp3"
                __text_to_speech__(baxter_response, response_recording)
                __display_sound__(response_recording)
                state_counter += 1
                print("In state: ", state_counter - 1)
            elif((state_counter == 4) & (lower_case.__contains__("thank you"))):
                command = "rosrun baxter_examples joint_trajectory_file_playback.py -f ./trajectories/baxter_farewell.csv"
                call_wait = subprocess.Popen(["python", "./execute_remote.py", command], shell=True)
                call_wait.wait()
                baxter_response = f"It was a pleasure serving you ." #TODO:erase
                response_recording = "./sound_files/baxter_farewell.mp3" #TODO:erase
                __text_to_speech__(baxter_response, response_recording)
                __display_sound__(response_recording)
                time.sleep(10)
                print("In state: ", state_counter - 1)
                state_counter = 0
        except sr.UnknownValueError:
            baxter_response = f"Sorry, I didn't understand that." #TODO:erase
            response_recording = "./sound_files/non-comprehending_baxter.mp3" #TODO:erase
            __text_to_speech__(baxter_response, response_recording) #TODO:erase
            __display_sound__(response_recording)
        except sr.RequestError:
            baxter_response = f"Sorry, something went wrong with the API request." #TODO:erase
            response_recording = "./sound_files/baxter_api_error.mp3" #TODO:erase
            __text_to_speech__(baxter_response, response_recording)
            __display_sound__(response_recording)


