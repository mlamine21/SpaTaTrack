import time
import sys
# import fluidsynth
from mingus.midi import midi_file_in
from mingus.midi import fluidsynth
import threading
from  mingus.containers.instrument import MidiInstrument


class LoopThread(threading.Thread):
    def __init__(self, track=None, channel=0):
        super().__init__()
        self.track = track
        self.channel = channel
    def run(self):
        if len(self.track)>0 :
            print(self.track)
            for beat in self.track:
                # print(beat)
                fluidsynth.play_NoteContainer(beat[2], self.channel)
                time.sleep((bar.meter[0]/beat[1]) * 60 / bpm)
                fluidsynth.stop_NoteContainer(beat[2], self.channel)

# fs = fluidsynth.Synth(gain=0.5)

# fs.start()
# fs.setting("synth.gain", 5.0)
# sfid = fs.sfload(r'FluidR3_GM.sf2')  # replace path as needed
# fs.program_select(0, sfid, 0, 0)
# fs.start(driver = 'dsound')  # use DirectSound driver
fluidsynth.init('GeneralUser-GS.sf2', 'dsound')  # replace path as needed

print("Synthesizer started")


composition, bpm = midi_file_in.MIDI_to_Composition("The-Final-Countdown.mid")
print("MIDI file loaded, BPM:", bpm)
print("Number of tracks:", len(composition.tracks))
print("Title : ", composition.title)

idx=0
notes_to_play = {}
for idx, track in enumerate(composition.tracks):
    if(track[0] is not None):
        notes_to_play[track.name+"_"+str(idx) ] = {"notes":[], "instrument": track.instrument}
        if track.instrument is not None:
            fluidsynth.set_instrument(idx,track.instrument)
            print("Track name:", track.name, "Instrument:", track.instrument, idx, MidiInstrument().names[track.instrument])
            notes_to_play[track.name+"_"+str(idx)]["channel"] = idx
# tracks = composition.tracks
# #remove empty tracks
# tracks = [track for track in tracks if track[0] is not None]

# notes_to_play = {t.name+"_"+str(i): {"notes":[], "instrument": t.instrument} for i, t in enumerate(tracks)}
names = {t.name : t.name+"_"+str(i) for i, t in enumerate(composition.tracks)}
print(names)
length_max=1
notes_length = 1
# notes_length = 1/bar.meter[0]
#Decompose each track into bars then bars into 1/4 bars (beats) and store them in a list of lists
for track in composition.tracks:
    for bar in track:
        if bar is not None:
            # Decompose each bar into 1/4 bars (beats)
            beats = []
            idx=0
            if(len(bar)==0 or len(bar[0][-1])==0):
                # print("Bar is empty")
                continue
            length_total = 0
            # Decompose each bar into 1/4 bars (beats)
            # Start from the first beat and keep adding beats until we reach 1/4 of a bar (or the end of the bar)
            while(length_total<length_max and len(bar)>idx):
                length=0
                beats = []
                while(len(bar)>idx and length < notes_length):
                    # print(length, bar.meter[0]/bar[idx][1])
                    beats.append(bar[idx])
                    length += 1/bar[idx][1]
                    idx+=1
                length_total += length
                # Notes to play contains for each track a list of lists of beats (1/4 bars) to play
                notes_to_play[names[track.name]]["notes"].append(beats)

val = input("Press Enter to start playing the first 1/4 bar of each track...")

idx=0
while(val!="q"):
    #Play the first 1/4 bar of each track (simultaneously) 
    #Start a thread for each track to play the first 1/4 bar of each track simultaneously
    threads = []
    channel = 0
    for key in notes_to_play.keys():
        
        if(len(notes_to_play[key]["notes"])>idx):
            threads.append(LoopThread(track=notes_to_play[key]["notes"][idx], channel=notes_to_play[key]["channel"]))
        channel+=1
    for thread in threads:
        thread.start()
    if(len(threads)>0):
        threads[0].join()  # Wait for the last thread to finish (all threads should finish at the same time but with a little bit of delay)
    idx += 1
    val = input("Press Enter to play the next 1/4 bar of each track, or 'q' to quit...")

# while(val!="q"):
#     bars_to_play = []
#     for track in tracks:
#         if len(track.bars) <= bar_num:
#             print("Track", track.name, "has no more bars to play.")
#             continue
#         else:
#             print("Playing bar", bar_num, "of track", track.name)
#             bars_to_play.append(track.bars[bar_num])


#     #Play the first bar of each track (simultaneously) 
#     #Start a thread for each track to play the first bar of each track simultaneously
#     threads = []
#     for bar in bars_to_play:
#         threads.append(LoopThread(bar=bar))

#     for thread in threads:
#         thread.start()
#     if(len(threads)>0):
#         threads[-1].join()  # Wait for the last thread to finish (all threads should finish at the same time but with a little bit of delay)

#     bar_num += 1
#     val = input("Press Enter to play the next bar of each track, or 'q' to quit...")
#Extract beats so that we can play for 1/4 a bar

# for bar in track.bars:
#     print(bar.meter)
    # for beat in bar:
    #     # print(beat)
    #     fluidsynth.play_NoteContainer(beat[2])
    #     time.sleep((bar.meter[0]/beat[1]) * 60 / bpm)  # Sleep for the duration of the beat in seconds
        # for note in track.get_notes():
        #     print(note)
        #     print(note[2])
        #     fluidsynth.play_NoteContainer(note[2], bpm)
        #     time.sleep(1/note[1] * 60 / bpm)  # Sleep for the duration of the note in seconds
    # for bar in track.bar:
    #     for beat in bar:
    #         print(beat)
    #         fs.noteon(0, beat.pitch, beat.velocity)

# fs.noteon(0, 60, 50) # channel, pitch, velocity

# time.sleep(1.0)

# fs.noteoff(0, 60)

# time.sleep(1.0)

# fs.delete()