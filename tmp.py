import pyttsx3

# Create a new text-to-speech engine
engine = pyttsx3.init()

# Set the voice
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id) # Change the index number to select a different voice

# Set the speed (optional)
engine.setProperty('rate', 200) # Change the value to adjust the speed

# Enter the text to be spoken
text = "Hello, world! mh17av2939 is an example text."
txt = "CZ19923"

engine.say(text)
for each in txt:
    # Speak the text
    engine.say(each)
    engine.runAndWait()