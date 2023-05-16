import speech_recognition as sr
from contextlib import suppress

with suppress(Exception):

    # set up the recognizer
    r = sr.Recognizer()

    # set up the microphone
    mic = sr.Microphone()

    # set the keyword we want to detect
    keyword = "coffee"
    keywords = [keyword, "cappuccino", "latte", "espresso", "mocha", "americano", "macchiato", "cortado", "affogato", "flat white", "frappe", "irish coffee", "ristretto", "lungo", "ristretto", "doppio"]

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
            if any(key in text.lower() for key in keywords):
            # if keyword in text.lower():
                # do something here, like play a sound or send a notification
                # os.system("aplay /usr/share/sounds/alsa/Front_Center.wav")
                with open("other_code.py", "r") as f:
                    exec(f.read())
                print("Detected keyword: " + keyword)

        except sr.UnknownValueError:
            print("Sorry, I didn't understand that.")
        except sr.RequestError:
            print("Sorry, something went wrong with the API request.")
