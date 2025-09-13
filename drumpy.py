#!/usr/bin/env python
#
"""

"""


import functools
import logging
import sys
import time
import numpy as np
import queue
import threading
import plotille
from argparse import ArgumentParser
from pathlib import Path
from icecream import ic


from rtmidi.midiutil import open_midiinput
from wakepy import keep

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

log = logging.getLogger('midiin_poll')
logging.basicConfig()


timestamps = queue.Queue()
x1 = queue.Queue()

translate = {
        38: 2.5, # Snare
        36: 0.5, # Basso
        43: 1.5, # floor-tom
        45: 2.0, # mid-tom
        48: 3.5, # high tom
        46: 4.5, # hihat
        42: 4.5, # hihat
        22: 4.5, # hihat
        51: 5.5,  # ride
        }



def data_generator(args):
    # Prompts user for MIDI input port, unless a valid port number or name
    # is given as the first argument on the command line.
    # API backend defaults to ALSA on Linux.
    port = args.port
    try:
        midiin, port_name = open_midiinput(port)
    except (EOFError, KeyboardInterrupt):
        sys.exit()

    print("Entering main loop. Press Control-C to exit.")
    i = 0
    t0 = None
    tlast = None
    try:
        timer = time.time()
        while True:
            i += 1
            msg = midiin.get_message()
            t = time.time()

            if msg:
                message, deltatime = msg
                timer += deltatime
                log.debug("[%s] @%0.6f %r" % (port_name, timer, message))
                if message[0] == 153:
                    if t0 is None:
                        t0 = t
                    if message[1] in translate:
                        timestamps.put(time.time() - t0)
                        x1.put(translate[message[1]])
                    if tlast is not None and t - tlast > 10:
                        print("Reset start time")
                        t0 = t
                    tlast = t
            time.sleep(0.01)
    except KeyboardInterrupt:
        print('')
    finally:
        plt.close()
        print("Exit.")
        midiin.close_port()
    del midiin

def get_tempo(t, y, args):
    """
    Calculates tempo assuming that there are 2 hihat or ride hits per beat

    Rounds the end results to 10bpm
    """
    hihats_ride = t[(y == 4.5) | (y == 5.5)]
    try:
        if len(hihats_ride) < 5:
            return None
    except:
        return None
    dt = (hihats_ride[1:] - hihats_ride[:-1])
    dt = dt[(dt > 0.1) * (dt < 2)]
    perc = np.percentile(dt, [10, 90])
    dt = dt[(dt > perc[0]) * (dt < perc[1])]
    dt = dt.mean()
    T = dt * args.tmp
    bps = 1 / T
    bpm = bps * 60
    return np.round(bpm, decimals=-1)


def test(args):
    """
    Produces test data and draws the visualization with that
    """
    beats = args.bar
    hihat = np.arange(0, beats, 0.5)
    bass = np.arange(0, beats, 2)
    snare = np.arange(0, beats, 2) + 1
    xx = np.concatenate([
        hihat,
        bass,
        snare,
        ])
    yy = np.concatenate([
        np.ones_like(hihat) * 4.5,
        np.ones_like(bass) * 0.5,
        np.ones_like(snare) * 2.5,
        ])

    yy = np.repeat(yy, 10)
    xx = np.repeat(xx, 10) + np.random.normal(0.0, scale=0.04, size=10 * len(xx))

    draw_plotille(xx, yy, bar=args.bar)

def draw_plotille(xx, y, bar=4):
    """
    Draws the plotille visualization
    """
    fig = plotille.Figure()
    fig.width = 90
    fig.height = 12
    fig.color_mode = 'byte'
    for i in np.arange(0, bar, 0.25):
        fig.plot([i,i], [0,5], lc=8)
    fig.scatter(xx, y, lc=200)
    fig.scatter(xx - bar, y, lc=25)
    fig.set_x_limits(min_=-0.5, max_=bar)
    fig.set_y_limits(min_=0, max_=6)
    print(fig.show())

    
def main():
    parser = ArgumentParser(
            prog='',
            description='TODO',
            )
    parser.add_argument("-p", "--port", type=int, help="MIDI port to use")
    parser.add_argument("--test", action="store_true", help="Test mode for the visualization. Not interactive")
    parser.add_argument("--plotille", action="store_true", help="Use plotille instead of pyplot")
    parser.add_argument("-b", "--bpm", type=int, help="BPM to use. Leave empty to use tempo detection")
    parser.add_argument("--bar", type=int, help="Beats in a bar", default=4)
    parser.add_argument("--sub", type=int, help="Beat subdivision, Options are 2, 3, 4. Default 4", default=4)
    parser.add_argument("-o", "--output", type=Path, help="Create a video file. Note: You won't see the visualization")
    args = parser.parse_args()

    if args.test:
        test(args)
        return
    threading.Thread(target=data_generator, args=[args], daemon=True).start()

    ic(args.bpm)
    bpm = args.bpm
    if bpm is None:
        bpm = 100
    bps = bpm / 60

    if args.sub is None:
        args.sub = 4

    if args.sub != 3:
        args.tmp = 2
    else:
        args.tmp = 3

    x_data, y_data = [], []
    if not args.plotille:
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(15,10))
        line = ax.scatter([], [], ls="", marker="o", c=[], s=250, cmap="Greys", alpha=1, vmin=0, vmax=1)
        line2 = ax.scatter([], [], ls="", marker="o", c=[], s=250, cmap="Reds", alpha=1, vmin=0, vmax=1)
        # Set up the plot's appearance
        ax.set_xlim(-0.5, args.bar + 0.5)  # Initial x-axis limits
        ax.set_ylim(-1.5, 6.5)  # Initial y-axis limits
        ax.set_title("Real-Time Data Plot")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        match args.sub:
            case 4:
                xticks = np.linspace(0, args.bar, args.bar * 4 + 1)
                tick_labels = list("1e&a2e&a3e&a4e&a5e&a6e&a")[:len(xticks)-1] + ["1"]
            case 2:
                xticks = np.linspace(0, args.bar, args.bar * 2 + 1)
                tick_labels = list("1&2&3&4&5&6&")[:len(xticks)-1] + ["1"]
            case 3:
                xticks = np.linspace(0, args.bar, args.bar * 3 + 1)
                tick_labels = list("1&a2&a3&a4&a5&a6&a")[:len(xticks)-1] + ["1"]
            case _:
                raise ValueError("Unknown subdivision")
        ax.set_xticks(xticks, tick_labels)

        ax.grid(True)
    else:
        fig = None


    # Initialize the line (called at the start of the animation)
    def init():
        line.set_offsets(np.empty((0,2)))
        line2.set_offsets(np.empty((0,2)))
        line.set_array(np.array([]))
        line2.set_array(np.array([]))
        return [line, line2]

    def update_plotille():
        while not timestamps.empty():
            x_data.append(timestamps.get())
            y_data.append(x1.get())

            # Keep only the last 100 points for performance
            if len(x_data) > 100:
                x_data.pop(0)
                y_data.pop(0)
        if len(x_data) == 0:
            return []
        if args.bpm is None:
            tempo = get_tempo(np.array(x_data), np.array(y_data), args)
        else:
            tempo = None

        if tempo is not None:
            a = tempo / 60
        else:
            a = bps
        print(f"{tempo=}")
        # Update the plot
        xx = (np.array(x_data) * a) % 4
        draw_plotille(xx, y_data)


    # Update function for the animation
    def update(frame):
        # Get data from the queue
        #print("Update")
        while not timestamps.empty():
            x_data.append(timestamps.get())
            y_data.append(x1.get())

            # Keep only the last 100 points for performance
            if len(x_data) > 100:
                x_data.pop(0)
                y_data.pop(0)

        if len(x_data) == 0:
            return []
        if args.bpm is None:
            tempo = get_tempo(np.array(x_data), np.array(y_data), args)
        else:
            tempo = None

        if tempo is not None:
            a = tempo / 60
        else:
            a = bps
        logging.debug(f"{tempo=}")
        # Update the plot
        xx = (np.array(x_data) * a) % 4
        line.set_offsets(list(zip(xx, y_data)))
        line2.set_offsets(list(zip(xx - 4, y_data)))
        i = 100 - np.arange(len(xx))[::-1]
        line.set_array(i / 100)
        line2.set_array(i / 100)
        return line, line2

    with keep.presenting():
        if not args.plotille:
            save_thread = None
            if fig is None:
                raise ValueError("figure hasn't been created")
            # Create the animation
            try:
                ani = FuncAnimation(
                    fig, update, init_func=init, blit=True, interval=1000/60, frames=300,
                )

                # Show the plot
                if args.output is not None:
                    writervideo = animation.FFMpegWriter(fps=60)

                    def save_animation():
                        ani.save(args.output, writer=writervideo)
                    save_thread = threading.Thread(target=save_animation)
                    save_thread.start()

                plt.show()

                if save_thread is not None:
                    save_thread.join()

            except KeyboardInterrupt:
                print('')
            finally:
                print("Exit.")
                plt.close()
                if save_thread is not None:
                    save_thread.join()
        else:
            while True:
                update_plotille()
                time.sleep(0.1)

if __name__ == "__main__":
    main()
