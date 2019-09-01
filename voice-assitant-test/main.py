from datetime import datetime, timedelta
import speech_recognition as sr
import argparse

parser = argparse.ArgumentParser(description="Pipeline for keywords predicts")
parser.add_argument("--wait", type=int, default="",help="Waiting time")
args = parser.parse_args()
wait = args.wait

recognizer = sr.Recognizer()
recognizer.energy_threshold = 7000
recognizer.non_speaking_duration = 0.3
recognizer.pause_threshold = 0.3
recognizer.dynamic_energy_adjustment_ratio = 2
microphone = sr.Microphone()
i = 0

next_time = datetime.now() - timedelta(seconds=1)
current_time = datetime.now().time()

def listen():
    global i
    with microphone as source:
        audio = recognizer.listen(source)

        print("audio", audio)

    # write audio to a WAV file
    filename = "microphone-results-" + str(i) + ".wav"
    i = i + 1
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())

while True:
    current_time = datetime.now()
    if( current_time >= next_time):
        listen()
        next_time = datetime.now() + timedelta(seconds=wait)
        print(next_time);