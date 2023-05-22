import speech_recognition as sr
from contextlib import suppress
import coffee_make as coffee
from gtts import gTTS
import pygame
import time


state_counter = 0
bool_name = False
#initiate coffee maker
coffee_maker = coffee.Coffee()



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
                baxter_response = "Hi there, so nice to have you here! Could you give me the honour of acquiring your name?"
                response_recording = "./sound_files/baxter_greeting.mp3"
                __text_to_speech__(baxter_response, response_recording)
                __display_sound__(response_recording)
                state_counter = state_counter + 1
                print("In state: ", state_counter - 1)
                bool_name = True

            elif(state_counter == 1):
                baxter_response = f"Welcome to the best cappuccino stand in the world {text}. I need to learn your sugar preference before I get started. Would you like low sugar coffee or high sugar coffee?"
                response_recording = "./sound_files/baxter_sugar_request.mp3"
                __text_to_speech__(baxter_response, response_recording)
                __display_sound__(response_recording)
                state_counter = state_counter + 1
                print("In state: ", state_counter - 1)
                bool_name = False

            elif((state_counter == 2)):
                if(text.__contains__("low sugar")):
                    baxter_response = "What a healthy choice congratulations! It will be ready in a minute."
                    response_recording = "./sound_files/baxter_preparation_initalization_low_milk.mp3"
                    __text_to_speech__(baxter_response, response_recording)
                    __display_sound__(response_recording)
                    state_counter = state_counter + 1
                    print("In state: ", state_counter - 1)
                    coffee_maker.prepare(low_sugar = True)
                    #TODO: Low sugar preparation
                elif(text.__contains__("high sugar")):
                    baxter_response = "Please beware, sugar is not that healthy a choice. It will be ready in a minute."
                    response_recording = "./sound_files/baxter_preparation_initalization_high_milk.mp3"
                    __text_to_speech__(baxter_response, response_recording)
                    __display_sound__(response_recording)
                    state_counter = state_counter + 1
                    coffee_maker.prepare(low_sugar = False)
                    print("In state: ", state_counter - 1)
                else:
                    baxter_response = "I couldn't undestand. Could you please say merely high sugar or low sugar?"
                    response_recording = "./sound_files/baxter_clarification_sugar.mp3"
                    __text_to_speech__(baxter_response, response_recording)
                    __display_sound__(response_recording)
                    print("In state: ", state_counter)
            elif(state_counter == 3):
                baxter_response = f"Here is your coffee dear {text}. It was a pleasure serving you."
                response_recording = "./sound_files/baxter_sugar_farewell.mp3"
                __text_to_speech__(baxter_response, response_recording)
                __display_sound__(response_recording)
                print("In state: ", state_counter - 1)
                state_counter = 0

        except sr.UnknownValueError:
            baxter_response = f"Sorry, I didn't understand that."
            response_recording = "./sound_files/non-comprehending_baxter.mp3"
            __text_to_speech__(baxter_response, response_recording)
            __display_sound__(response_recording)
        except sr.RequestError:
            baxter_response = f"Sorry, something went wrong with the API request."
            response_recording = "./sound_files/non-comprehending_baxter.mp3"
            __text_to_speech__(baxter_response, response_recording)
            __display_sound__(response_recording)


