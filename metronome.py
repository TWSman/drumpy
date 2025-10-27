"""
Basic metronome implementation with accurate timing using sounddevice.
Should produce beeps with 1 ms second accuracy.
"""

import time
import threading
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from icecream import ic


def main():
    m = Metronome(bpm=100, beats_per_measure=4)
    m.start()

    try:
        while True:
            user_input = input("Enter new BPM (or 'q' to quit): ")
            if user_input.lower() == 'q':
                break
            try:
                new_bpm = int(user_input)
                m.set_bpm(new_bpm)
                print(f"Tempo changed to {new_bpm} BPM")
            except ValueError:
                print("Please enter a valid number.")
    except KeyboardInterrupt:
        pass
    finally:
        m.stop()
        print("Metronome stopped.")

def generate_click(frequency=1000, duration=0.05, volume=0.5, samplerate=44100):
    """
    Generate a short sine-wave click sound.
    frequency: Hz
    duration: seconds
    volume: 0.0–1.0
    """
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    wave = volume * np.sin(2 * np.pi * frequency * t)
    return wave.astype(np.float32)


class MetronomeBasic:
    def __init__(self, bpm=120, beats_per_measure=4):
        self.bpm = bpm
        self.beats_per_measure = beats_per_measure
        self.running = False
        self.lock = threading.Lock()
        self.start_time = None

        # Pre-generate sounds
        self.high_click = generate_click(frequency=1500)  # Beat 1
        self.low_click = generate_click(frequency=1000)  # Other beats

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False

    def set_bpm(self, bpm):
        with self.lock:
            self.bpm = bpm

    def _run(self):
        beat = 1
        while self.running:
            with self.lock:
                bpm = self.bpm
            beat_duration = 60.0 / bpm
            if self.start_time is None:
                self.start_time = time.time()
                ic(self.start_time)

            if beat == 1:
                sd.play(self.high_click, samplerate=44100)
                print(f"Beat 1 (BPM: {bpm})")
            else:
                sd.play(self.low_click, samplerate=44100)
                print(f"Beat {beat} (BPM: {bpm})")

            sd.wait()
            time.sleep(max(0, beat_duration - 0.05))

            beat += 1
            if beat > self.beats_per_measure:
                beat = 1


def plot_results(t, bpm):
    fig, axs = plt.subplots(3,1,figsize=(10,10))
    x = np.array(t) - t[0]
    d = np.diff(x)
    for ax in axs[:2]:
        ax.plot(x[1:], d, marker='o', linestyle='None')
        ax.axhline(60.0/bpm, color='r', linestyle='--', label='Ideal Beat Interval')
    mn_val = 60.0 / bpm * 0.95
    mx_val = 60.0 / bpm * 1.05
    axs[2].hist(d, bins=np.linspace(mn_val, mx_val, 100), edgecolor='black')
    for ax in axs:
        ax.grid(True)
    axs[0].set_ylim(60.0/bpm * 0.95, 60.0/bpm * 1.05)
    axs[1].set_ylim(60.0/bpm * 0.99, 60.0/bpm * 1.01)
    plt.savefig("metronome_timing.png")

class Metronome:
    """
    Improved metronome. Should keep time within 1ms
    """
    def __init__(self, bpm=120, beats_per_measure=4, samplerate=44100, plot_results=False):
        self.bpm = bpm
        self.beats_per_measure = beats_per_measure
        self.running = False
        self.lock = threading.Lock()
        self.samplerate = samplerate
        self.plot = plot_results
        self.logs = []

        # Pre-generate sounds
        self.high_click = generate_click(frequency=1500, samplerate=samplerate)  # Beat 1
        self.low_click = generate_click(frequency=1000, samplerate=samplerate)   # Other beats

    def start(self):
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self.running = False
        if self.plot:
            plot_results(self.logs, self.bpm)

    def set_bpm(self, bpm):
        with self.lock:
            self.bpm = bpm

    def _run(self):
        beat = 1
        next_beat_time = time.perf_counter()
        while self.running:
            with self.lock:
                bpm = self.bpm
            beat_duration = 60.0 / bpm

            now = time.perf_counter()

            if now < next_beat_time:
                time.sleep(max(0, next_beat_time - now))

            self.logs.append(time.perf_counter())
            if beat == 1:
                sd.play(self.high_click, samplerate=44100)
                print(f"Beat 1 (BPM: {bpm})")
            else:
                sd.play(self.low_click, samplerate=44100)
                print(f"Beat {beat} (BPM: {bpm})")

            next_beat_time += beat_duration

            beat += 1
            if beat > self.beats_per_measure:
                beat = 1


class MetronomeSampling:
    """
    Should be a sample perfect metronome
    """
    def __init__(self, bpm=120, beats_per_measure=4, samplerate=44100):
        self.samplerate = samplerate
        self.bpm = bpm
        self.beats_per_measure = beats_per_measure
        self.samples_per_beat = int(self.samplerate * 60.0 / self.bpm)
        self.logs = []

        # Generate sounds
        self.high_click = generate_click(1500, samplerate=samplerate)
        self.low_click = generate_click(1000, samplerate=samplerate)

        self.beat = 1
        self.sample_counter = 0
        self.lock = threading.Lock()
        self.running = False
        self.playing = []

        self.beat_times = []
        self.beat_sample_positions = []
        self.total_samples = 0

        # Audio stream
        self.stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=1,
            callback=self._callback,
            blocksize=0,  # Let sounddevice choose optimal buffer size
        )

    def set_bpm(self, bpm):
        with self.lock:
            self.bpm = bpm
            self.samples_per_beat = int(self.samplerate * 60.0 / self.bpm)

    def start(self):
        self.running = True
        self.stream.start()
        print(f"Metronome started at {self.bpm} BPM")

    def stop(self):
        self.running = False
        time.sleep(0.1)
        self.stream.stop()
        print("Metronome stopped.")
        sample_times = [s / self.samplerate for s in self.beat_sample_positions]
        plot_results(sample_times, self.bpm)
        #plot_results(self.beat_times, self.bpm)

    def _callback(self, outdata, frames, time_info, status):
        """Fill the output buffer with silence or click sounds as needed."""
        out = np.zeros(frames, dtype=np.float32)
        idx = 0

        new_playing = []
        for click, pos in self.playing:
            end = pos + frames 
            if end < len(click):
                out += click[pos:end]
                new_playing.append((click, end))
            else:
                    out[:len(click) - pos] += click[pos:]

        self.playing = new_playing

        while idx < frames:
            if self.sample_counter == 0 and self.running:
                self.beat_times.append(time.perf_counter())
                self.beat_sample_positions.append(self.total_samples + self.sample_counter)
                # Trigger a click exactly on the beat boundary
                if self.beat == 1:
                    click = self.high_click
                else:
                    click = self.low_click

                self.playing.append((click, 0))

                self.beat += 1
                if self.beat > self.beats_per_measure:
                    self.beat = 1

            # Move forward
            step = min(frames - idx, self.samples_per_beat - self.sample_counter)
            idx += step
            self.sample_counter = (self.sample_counter + step) % self.samples_per_beat

        for click, pos in self.playing:
            length = min(frames, len(click) - pos)
            if length > 0:
                out[:length] += click[pos:pos+length]
        outdata[:] = out.reshape(-1, 1)
        self.total_samples += frames


if __name__ == "__main__":
    main()

