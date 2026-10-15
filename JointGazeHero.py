from roomConfiguration import RoomConfiguration
import time
import sys
import os
import json
# import fluidsynth
from mingus.midi import midi_file_in
from mingus.midi import fluidsynth
import threading
from mingus.containers.instrument import MidiInstrument
# add these GUI imports
import tkinter as tk
from tkinter import ttk, messagebox

class JointGazeHero():
    notes_to_play = {}
    interaction_objects = []
    def __init__(self, master=None, room_config : RoomConfiguration = None, midi_dir = "midis", config_midi = "config_midi.json"):
        self.room_config = room_config if room_config is not None else RoomConfiguration()
        self.midi_dir = midi_dir
        self.config_midi = os.path.join(midi_dir, config_midi)
        self.get_interaction_objects()
        self.init_midi_player()

        # GUI-related setup (place in __init__ after init_midi_player())
        self._created_root = False
        if master is None:
            self.root = tk.Tk()
            self._created_root = True
            self.root.title("JointGazeHero")
            self.root.geometry("900x600")
        else:
            self.root = master

        # Frames
        self.container = ttk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.menu_frame = ttk.Frame(self.container)
        self.music_frame = ttk.Frame(self.container)
        self.game_frame = ttk.Frame(self.container)

        # game UI placeholders
        self.midi_listbox = None
        self.square_frames = []
        self.current_index = 0
        self.score = 0
        self.selected_midi = None

        # callbacks for integration (set by caller)
        self.on_request_input = None
        self.on_music_start = None
        self.on_music_stop = None

        # highlight / timeout jobs
        self._highlight_job = None
        self._input_timeout_job = None

        # build game widgets (score, squares container, instruction label, overlay)
        # Score
        self.score_var = tk.StringVar(value=str(self.score))
        self.score_label = ttk.Label(self.game_frame, textvariable=self.score_var, font=("Helvetica", 36))
        self.score_label.pack(side=tk.TOP, pady=10)
        # Squares container
        self.squares_container = ttk.Frame(self.game_frame)
        self.squares_container.pack(expand=True)
        # instruction label
        self.instruction_var = tk.StringVar(value="")
        self.instruction_label = ttk.Label(self.game_frame, textvariable=self.instruction_var, font=("Helvetica", 16))
        self.instruction_label.pack(side=tk.BOTTOM, pady=10)
        # countdown overlay
        self.overlay = tk.Label(self.root, text="", font=("Helvetica", 96), fg="white", bg="black")
        self.overlay.place_forget()

        # show initial menu
        self.show_menu()
    
    def get_midi_files(self):
        midi_files = [f for f in os.listdir(self.midi_dir) if f.endswith('.mid') or f.endswith('.midi')]
        return midi_files

    def get_interaction_objects(self):
        self.interaction_objects = []
        for name, params in self.room_config.objects.items():
            if "interaction" in name:
                params["name"] = name
                self.interaction_objects.append(params)            

    def init_midi_player(self):
        fluidsynth.init('FluidR3_GM.sf2', 'dsound')  # replace path as needed
        print("Synthesizer started")
    
    def init_midi_composition(self, midi_file):
        
        composition, bpm = midi_file_in.MIDI_to_Composition(os.path.join(self.midi_dir,midi_file))
        idx=0
        self.notes_to_play = {}
        for idx, track in enumerate(composition.tracks):
            if(track[0] is not None):
                self.notes_to_play[track.name+"_"+str(idx) ] = {"notes":[], "instrument": track.instrument}
                if track.instrument is not None:
                    fluidsynth.set_instrument(idx,track.instrument)
                    print("Track name:", track.name, "Instrument:", track.instrument, idx, MidiInstrument().names[track.instrument])
                    self.notes_to_play[track.name+"_"+str(idx)]["channel"] = idx
        
        names = {t.name : t.name+"_"+str(i) for i, t in enumerate(composition.tracks)}
        #Open config_midi.json and read the "notes_length" parameter for the midi file
        with open(self.config_midi, "r") as f:
            config = json.load(f)
        notes_length = config.get(midi_file).get("notes_length")

        #Decompose each track into bars then bars into notes_length (nth beats) and store them in a list of lists
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
                    while(length_total<1 and len(bar)>idx):
                        length=0
                        beats = []
                        while(len(bar)>idx and length < notes_length):
                            # print(length, bar.meter[0]/bar[idx][1])
                            beats.append(bar[idx])
                            length += 1/bar[idx][1]
                            idx+=1
                        length_total += length
                        # Notes to play contains for each track a list of lists of beats (1/4 bars) to play
                        self.notes_to_play[names[track.name]]["notes"].append(beats)

    def show_menu(self):
        for w in self.container.winfo_children():
            w.pack_forget()

        self.menu_frame.pack(fill=tk.BOTH, expand=True)
        for c in self.menu_frame.winfo_children():
            c.destroy()

        title = ttk.Label(self.menu_frame, text="JointGazeHero", font=("Helvetica", 32))
        title.pack(pady=40)

        start_btn = ttk.Button(self.menu_frame, text="Commencer", command=self.show_music_selection)
        start_btn.pack(pady=10)

        quit_btn = ttk.Button(self.menu_frame, text="Quitter", command=self._quit)
        quit_btn.pack(pady=10)

    def show_music_selection(self):
        for w in self.container.winfo_children():
            w.pack_forget()

        self.music_frame.pack(fill=tk.BOTH, expand=True)
        for c in self.music_frame.winfo_children():
            c.destroy()

        label = ttk.Label(self.music_frame, text="Choisissez une musique", font=("Helvetica", 20))
        label.pack(pady=10)

        listbox_frame = ttk.Frame(self.music_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.midi_listbox = tk.Listbox(listbox_frame, height=10)
        self.midi_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL, command=self.midi_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.midi_listbox.config(yscrollcommand=scrollbar.set)

        # populate listbox with midi files
        try:
            midi_files = self.get_midi_files()
        except Exception:
            midi_files = []
        for f in midi_files:
            self.midi_listbox.insert(tk.END, f)

        btn_frame = ttk.Frame(self.music_frame)
        btn_frame.pack(pady=10)

        back_btn = ttk.Button(btn_frame, text="Retour", command=self.show_menu)
        back_btn.pack(side=tk.LEFT, padx=5)

        start_btn = ttk.Button(btn_frame, text="Start", command=self._start_selected_music)
        start_btn.pack(side=tk.LEFT, padx=5)

    def _start_selected_music(self):
        if not self.midi_listbox:
            return
        sel = self.midi_listbox.curselection()
        if not sel:
            messagebox.showinfo("Info", "Veuillez sélectionner une musique")
            return
        midi_file = self.midi_listbox.get(sel[0])
        print("Selected MIDI:", midi_file)
        # remember selected midi path
        self.selected_midi = os.path.join(self.midi_dir, midi_file)

        # load composition in background so UI stays responsive
        def load_comp():
            try:
                self.init_midi_composition(midi_file)
            except Exception as e:
                print("Error initializing MIDI composition:", e)
        t = threading.Thread(target=load_comp, daemon=True)
        t.start()

        # prepare and show the game screen immediately
        self._prepare_game_screen()
        self.show_game_screen()

        # start countdown and then the game flow
        self.start_countdown(self._after_countdown_start)

    def run(self):
        if self._created_root:
            self.root.mainloop()

    def _quit(self):
        if self._created_root:
            try:
                self.root.destroy()
            except Exception:
                pass
        else:
            try:
                self.root.quit()
            except Exception:
                pass
    def clamp(x): 
        return max(0, min(x, 255))    
    # --------------------- Game screen and flow methods ---------------------
    def _prepare_game_screen(self):
        # Build square widgets according to interaction_objects
        for child in self.squares_container.winfo_children():
            child.destroy()
        self.square_frames = []
        if not self.interaction_objects:
            self.interaction_objects = [{"name": "interaction_1", "color": "#dddddd"}]
        for i, obj in enumerate(self.interaction_objects):
            f = tk.Frame(self.squares_container, width=140, height=140, highlightthickness=2, relief=tk.RIDGE)
            f.pack(side=tk.LEFT, expand=True, padx=10, pady=20)
            f.pack_propagate(False)
            name = obj.get("name", f"interaction_{i}")
            color = obj.get("color", "#cccccc")
            print(name , color, obj)
            print(color)
            # normalize color: accept (r,g,b) tuples/lists or hex strings
            if isinstance(color, (tuple, list)) and len(color) == 3:
                try:
                    r, g, b = color
                    color = "#{:02x}{:02x}{:02x}".format(self.clamp(int(r)), self.clamp(int(g)), self.clamp(int(b)))
                except Exception:
                    color = "#cccccc"
            elif isinstance(color, str):
                # ensure it starts with # when it's a hex string (basic check)
                if color and not color.startswith("#"):
                    # try to accept names like 'white' by leaving as-is; tk will handle named colors
                    pass
            else:
                color = "#cccccc"

            lbl = tk.Label(f, text=name, bg=color)
            lbl.pack(fill=tk.BOTH, expand=True)
            self.square_frames.append((f, lbl))

    def show_game_screen(self):
        for w in self.container.winfo_children():
            w.pack_forget()
        self.game_frame.pack(fill=tk.BOTH, expand=True)

    def set_score(self, value:int):
        self.score = value
        try:
            self.score_var.set(str(self.score))
        except Exception:
            pass

    def highlight(self, index:int):
        self.clear_highlight()
        if index < 0 or index >= len(self.square_frames):
            return
        frame, lbl = self.square_frames[index]
        frame.config(highlightbackground="yellow", highlightthickness=6)
        # pulsing
        def pulse(on=True):
            try:
                frame.config(highlightbackground=("orange" if on else "yellow"))
                self._highlight_job = self.root.after(500, pulse, not on)
            except Exception:
                return
        pulse(True)

    def clear_highlight(self):
        if self._highlight_job:
            try:
                self.root.after_cancel(self._highlight_job)
            except Exception:
                pass
            self._highlight_job = None
        for frame, lbl in self.square_frames:
            try:
                frame.config(highlightbackground="black", highlightthickness=2)
            except Exception:
                pass

    def show_instruction(self, text:str, duration_ms:int=None):
        try:
            self.instruction_var.set(text)
            if duration_ms:
                self.root.after(duration_ms, lambda: self.instruction_var.set(""))
        except Exception:
            pass

    def start_countdown(self, callback_after_go=None):
        self.overlay.lift()
        self.overlay.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        seq = ["3", "2", "1", "GO"]
        def step(i=0):
            if i >= len(seq):
                self.overlay.place_forget()
                if callable(callback_after_go):
                    callback_after_go()
                return
            self.overlay.config(text=seq[i])
            self.root.after(800, step, i+1)
        step(0)

    def _after_countdown_start(self):
        # called after countdown; start music and first target
        if callable(self.on_music_start) and self.selected_midi:
            try:
                self.on_music_start(self.selected_midi)
            except Exception:
                pass
        self.current_index = 0
        if not self.square_frames:
            self._prepare_game_screen()
        self._start_target(self.current_index)

    def _start_target(self, index:int):
        self.highlight(index)
        # cancel previous timeout
        if self._input_timeout_job:
            try:
                self.root.after_cancel(self._input_timeout_job)
            except Exception:
                pass
            self._input_timeout_job = None
        # schedule timeout after 3 seconds
        self._input_timeout_job = self.root.after(3000, lambda: self.on_input_result(False))
        # notify external detector
        if callable(self.on_request_input):
            try:
                self.on_request_input(index)
            except Exception:
                pass

    def on_input_result(self, success:bool):
        # cancel timeout
        if self._input_timeout_job:
            try:
                self.root.after_cancel(self._input_timeout_job)
            except Exception:
                pass
            self._input_timeout_job = None

        if success:
            self.score += 1
            self.set_score(self.score)
            self.show_instruction("Well done", 800)
        else:
            self.show_instruction("Oops", 800)

        # move to next
        next_index = self.current_index + 1
        if next_index >= len(self.square_frames):
            self.clear_highlight()
            if callable(self.on_music_stop):
                try:
                    self.on_music_stop()
                except Exception:
                    pass
            self.show_instruction("Sequence finished", 2000)
            self.root.after(2000, self.show_menu)
            return
        self.current_index = next_index
        self.root.after(800, lambda: self._start_target(self.current_index))

if __name__ == "__main__":
    room_config = RoomConfiguration()

    joint_gaze_hero = JointGazeHero(room_config=room_config, midi_dir="midis", config_midi="config_midi.json")
    
    
    joint_gaze_hero.run()