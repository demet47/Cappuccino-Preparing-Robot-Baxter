import speech_recognition as sr
from contextlib import suppress
import coffee_make as coffee

state_counter = 0
bool_name = False
#initiate coffee maker
coffee_maker = coffee.Coffee()

coffee_maker.prepare(False)