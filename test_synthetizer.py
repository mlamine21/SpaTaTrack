import time
import fluidsynth

fs = fluidsynth.Synth(gain=0.5)

# fs.start()
fs.setting("synth.gain", 5.0)
sfid = fs.sfload(r'FluidR3_GM.sf2')  # replace path as needed
fs.program_select(0, sfid, 0, 0)
fs.start(driver = 'dsound')  # use DirectSound driver

fs.noteon(0, 100, 50) # channel, pitch, velocity

time.sleep(3.0)

fs.noteoff(0, 60)

time.sleep(1.0)

fs.delete()