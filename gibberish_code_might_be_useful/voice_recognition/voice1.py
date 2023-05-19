import speech_recognition as sr
import os

# set up the recognizer
r = sr.Recognizer()

# set up the microphone
mic = sr.Microphone()

# set the keyword we want to detect
keyword = "coffee"

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
        if keyword in text.lower():
            print("Detected keyword: " + keyword)
            # do something here, like play a sound or send a notification
            os.system("aplay /usr/share/sounds/alsa/Front_Center.wav")

    except sr.UnknownValueError:
        print("Sorry, I didn't understand that.")
    except sr.RequestError:
        print("Sorry, something went wrong with the API request.")
