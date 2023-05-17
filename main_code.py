import speech_recognition as sr
from contextlib import suppress
import pyttsx3
import coffee_make as coffee

state_counter = 0


def text_to_speech(text, output_file):
    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()

    # Save the speech output to a file
    engine.save_to_file(text, output_file)

    # Run the engine to process the text
    engine.runAndWait()

coffee_maker = coffee.Coffee()


with suppress(Exception):

    # set up the recognizer
    r = sr.Recognizer()

    # set up the microphone
    mic = sr.Microphone()

    # set the keyword we want to detect
  
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
            text = r.recognize_google(audio)
            print("You said: " + text)
            
            # check if the keyword is in the recognized text
            lower_case = text.lower()
            if((state_counter == 0) & (lower_case.__contains__("hello"))):
                baxter_response = "Hi there, so nice to have you here! Could you give me the honour of acquiring your name?"
                response_recording = "baxter_greeting.wav"
                text_to_speech(baxter_response, response_recording)
                state_counter = state_counter + 1
                print("In state: ", state_counter - 1)

            elif(state_counter == 1):
                baxter_response = f"Welcome to the best cappuccino stand in the world {text}. I need to learn your sugar preference before I get started. Would you like low sugar coffee or high sugar coffee?"
                response_recording = "baxter_sugar_request.wav"
                text_to_speech(baxter_response, response_recording)
                state_counter = state_counter + 1
                print("In state: ", state_counter - 1)

            elif((state_counter == 2)):
                if(text.__contains__("low sugar")):
                    baxter_response = "What a healthy choice congratulations! It will be ready in a minute."
                    response_recording = "baxter_preparation_initalization_low_milk.wav"
                    text_to_speech(baxter_response, response_recording)
                    state_counter = state_counter + 1
                    print("In state: ", state_counter - 1)
                    coffee_maker.prepare(low_sugar = True)
                    #TODO: Low sugar preparation
                elif(text.__contains__("high sugar")):
                    baxter_response = "Please beware, sugar is not that healthy a choice. It will be ready in a minute."
                    response_recording = "baxter_preparation_initalization_high_milk.wav"
                    text_to_speech(baxter_response, response_recording)
                    state_counter = state_counter + 1
                    coffee_maker.prepare(low_sugar = False)
                    print("In state: ", state_counter - 1)
                else:
                    baxter_response = "I couldn't undestand. Could you please say merely high sugar or low sugar?"
                    response_recording = "baxter_clarification_sugar.wav"
                    text_to_speech(baxter_response, response_recording)
                    print("In state: ", state_counter)


        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
        except sr.RequestError:
            print("Sorry, something went wrong with the API request.")
