import speech_recognition as sr
from contextlib import suppress
import coffee_make as coffee
from gtts import gTTS
import pygame
import time
import subprocess

state_counter = 2
bool_name = False
#initiate coffee maker
coffee_maker = coffee.Coffee()
name = ""

'''
VOICE FILES HARDCODED:
- ./sound_files/baxter_greeting.mp3
- ./sound_files/baxter_preparation_initalization_low_milk.mp3
- ./sound_files/baxter_preparation_initalization_high_milk.mp3
- ./sound_files/baxter_clarification_sugar.mp3
- ./sound_files/non-comprehending_baxter.mp3"
- ./sound_files/baxter_api_error.mp
- ./sound_files/mixer_request_from_baxter.mp3
- ./sound_files/baxter_farewell.mp3



TRAJECTORY FILES HARDCODED:
- ../trajectories/baxter_greet.csv
- ../trajectories/baxter_farewell.csv
- ../trajectories/low_sugar_part_1.csv
- ../trajectories/low_sugar_part_2.csv
- ../trajectories/high_sugar_part_1.csv
- ../trajectories/high_sugar_part_2.csv
PS: *_sugar_part_2.csv files include the serving gesture at the end, mixer grip and util at the beggining


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
                
                state_counter = state_counter + 1
                print("In state: ", state_counter - 1)
                bool_name = True

            elif(state_counter == 1):
                name = text
                baxter_response = f"Welcome to the best cappuccino stand in the world {name}. I need to learn your sugar preference before I get started. Would you like low sugar coffee or high sugar coffee?"
                response_recording = "./sound_files/baxter_sugar_request.mp3"
                __text_to_speech__(baxter_response, response_recording)
                __display_sound__(response_recording)
                state_counter = state_counter + 1
                print("In state: ", state_counter - 1)
                command = "cd alper_workspace; source activate_env.sh; rosrun baxter_examples joint_trajectory_file_playback.py -f ../trajectories/baxter_greet.csv"
                subprocess.call(["python", "./execute_remote.py", command], shell=True)
                bool_name = False

            elif((state_counter == 2)):
                if(text.__contains__("low sugar")):
                    baxter_response = "What a healthy choice congratulations! It will be ready in a minute." #TODO:erase
                    response_recording = "./sound_files/baxter_preparation_initalization_low_milk.mp3" #TODO:erase
                    __text_to_speech__(baxter_response, response_recording) #TODO:erase
                    __display_sound__(response_recording)
                    state_counter = state_counter + 1
                    print("In state: ", state_counter - 1)
                    coffee_maker.prepare(low_sugar = True)
                    #TODO: Low sugar preparation
                elif(text.__contains__("high sugar")):
                    baxter_response = "Please beware, sugar is not that healthy a choice. It will be ready in a minute." #TODO:erase
                    response_recording = "./sound_files/baxter_preparation_initalization_high_milk.mp3" #TODO:erase
                    __text_to_speech__(baxter_response, response_recording) #TODO:erase
                    __display_sound__(response_recording)
                    state_counter = state_counter + 1
                    coffee_maker.prepare(low_sugar = False)
                    print("In state: ", state_counter - 1)
                else:
                    baxter_response = "I couldn't undestand. Could you please say merely high sugar or low sugar?" #TODO:erase
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
                command = "cd alper_workspace; source activate_env.sh; nohup rosrun baxter_examples joint_trajectory_file_playback.py -f ../trajectories/baxter_farewell.csv"
                subprocess.call(["python", "./execute_remote.py", command], shell=True)
                baxter_response = f"It was a pleasure serving you ." #TODO:erase
                response_recording = "./sound_files/baxter_farewell.mp3" #TODO:erase
                __text_to_speech__(baxter_response, response_recording)
                __display_sound__(response_recording)
                time.sleep(10)
                state_counter += 1
                print("In state: ", state_counter - 1)
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


