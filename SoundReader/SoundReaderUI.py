"""
Sound Reader UI

A user interface for recording two voices saying the same words and displaying spectrograms for comparison.

Artin Kim
Adin Ackerman
"""

import os
from threading import Thread
from time import sleep

import tkinter as tk
import tkinter.ttk as ttk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import librosa
import pyaudio
import wave

from typing import Callable

# def file_handler_decorator(function):
#         def wrapper(*args, **kwargs):
#             try:
#                 function(*args, **kwargs)
#             except FileNotFoundError:
#                 print("No reference found") # no action is needed
#             except Exception as e:
#                 raise NotImplementedError(f'Error: {e}\nHandle all other errors.')
#         return wrapper

def threading_decorator(function: Callable) -> None:
    def wrapper(*args, **kwargs):
        Thread(target = function, daemon = True, args = args, kwargs = kwargs).start()
    return wrapper

class MainWindow(ttk.Frame):
    """
    The object wrapping the UI and logic, inherits ttk.Frame.

    Attributes
    ----------

    words: list[str]
        A list of words.
    word_index: int
        The index of the current word.
    reference_audio_path: str
        The path to the directory of prerecorded audio for each word.
    recording_audio_path: str
        The path the directory of currently recorded audio for each word.
    """

    words: list[str]
    word_index: int = 0
    reference_audio_path: str = 'Reference'
    recording_audio_path: str = 'Recording'

    def __init__(self, parent, *args, **kwargs):
        ttk.Frame.__init__(self, parent, *args, **kwargs)

        self.words = self.loadText('1000words.txt')

        if not os.path.isdir(self.recording_audio_path):
            os.mkdir(self.recording_audio_path)
        
        if not os.path.isdir(self.reference_audio_path):
            os.mkdir(self.reference_audio_path)
        
        self.p = pyaudio.PyAudio()
        self.stream_in = self.p.open(format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                output=False,
                input=True)
        self.stream_out = self.p.open(format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                output=True,
                input=False)

        self.parent = parent
        self.pack(fill = tk.BOTH, expand=True)
        
        top_frame = ttk.Frame(self)
        top_frame.pack(side = tk.TOP, fill = tk.BOTH, padx = 10, pady = 10)
        
        mid_frame = ttk.Frame(self)
        mid_frame.pack(side = tk.TOP, fill = tk.BOTH, padx = 10, pady = 10, expand = True)
        
        bot_frame = ttk.Frame(self)
        bot_frame.pack(side = tk.BOTTOM, fill = tk.BOTH, padx = 10, pady = 10)

        self.text_var = tk.StringVar()
        self.text_var.set(self.words[self.word_index])

        text = ttk.Label(top_frame, textvariable = self.text_var)
        text.pack(padx = 10, pady = 10)

        next_buttonn = ttk.Button(bot_frame, text = 'Next',command = self.nextWord)
        next_buttonn.pack(side = tk.RIGHT, padx = 10, pady = 10)
        prev_button = ttk.Button(bot_frame, text = 'Previous', command = self.previousWord)
        prev_button.pack(side = tk.LEFT, padx = 10, pady = 10)
        
        self.record_var = tk.StringVar()
        self.record_var.set("Record")
        replayReferenceButton = ttk.Button(mid_frame, text = 'Play Reference', command = self.playReference)
        replayReferenceButton.pack(side = tk.LEFT, padx = 10, pady = 10)
        self.recordButton = ttk.Button(mid_frame, textvariable = self.record_var, command = self.record)
        self.recordButton.pack(side = tk.LEFT, padx = 10, pady = 10)
        replayOwnButton = ttk.Button(mid_frame, text = 'Play Your Voice', command = self.playRecording)
        replayOwnButton.pack(side = tk.LEFT, padx = 10, pady = 10)
        # playBothButton = ttk.Button(mid_frame, text = 'Play Both')
        # playBothButton.pack(side = tk.LEFT, padx = 10, pady = 10)

        # fig = matplotlib.pyplot.specgram()
        fig = matplotlib.figure.Figure(figsize=(4, 3), dpi=100)
       
        self.subplotReference = fig.add_subplot(121)
        self.subplotRecording = fig.add_subplot(122)
        self.subplotReference.axis("off")
        self.subplotRecording.axis("off")
        self.subplotReference.set_title('Reference Voice')
        self.subplotRecording.set_title('Your Voice')
        
        # fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master = top_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill = tk.BOTH, padx = 10, pady = 10)
        
        self.update_reference_plot()
        self.update_recording_plot()

        
        # toolbar = NavigationToolbar2Tk(canvas, top_frame)
        # toolbar.update()
        # canvas.get_tk_widget().pack(side= tk.LEFT, padx = 10, pady = 10)

        
    def nextWord(self):
        self.word_index += 1
        self.text_var.set(self.words[self.word_index])
        self.update_reference_plot()
        self.update_recording_plot()

    def previousWord(self):
        self.word_index -= 1
        self.text_var.set(self.words[self.word_index])
        self.update_reference_plot()
        self.update_recording_plot()
    
    @staticmethod
    def loadText(file_name):
        with open(file_name, 'r') as f: 
            return f.readlines()

    # @file_handler_decorator
    @threading_decorator
    def playFile(self,file_name):
        try:
            with wave.open(file_name) as wf:
                print(wf.getnchannels(),wf.getframerate(),self.p.get_format_from_width(wf.getsampwidth()))
                data = wf.readframes(1024)
                while data:
                    self.stream_out.write(data,1024)
                    data = wf.readframes(1024)
        except FileNotFoundError:
            print(f'Can not find {file_name} audio file') # no action is needed
        except Exception as e:
            raise NotImplementedError(f'Error: {e}\nHandle all other errors.')
    

    def playReference(self):
        self.playFile(f'{self.reference_audio_path}/{self.word_index}.wav')

    def playRecording(self):
        self.playFile(f'{self.recording_audio_path}/{self.word_index}.wav')
            
    @threading_decorator        
    def record(self):
        self.record_var.set("Recording")
        self.recordButton.config(command = lambda: None)
        self.stream_in.start_stream()
        
        recording_frames = (self.stream_in.read(1024) for _ in range(44100 // (1024 * 1)))

        file_name = f'{self.recording_audio_path}/{self.word_index}.wav'

        open(file_name,'w').close()

        with wave.open(file_name,'wb') as wf:
            wf.setframerate(44100)
            wf.setnchannels(1)
            wf.setsampwidth(2) # 2 Bytes per sample (pyaudio.paInt16)
            wf.writeframes(b''.join(recording_frames))
        
        self.playRecording()
        self.update_recording_plot()

        self.stream_in.stop_stream()
        self.record_var.set("Record")
        self.recordButton.config(command = self.record)

   
    def update_reference_plot(self):
        try:
            y, sr = librosa.load(f'{self.reference_audio_path}/{self.word_index}.wav')
            self.subplotReference.specgram(y,Fs=512)
        except FileNotFoundError:
            print(f'No {self.reference_audio_path} audio found') # no action is needed
            self.subplotReference.clear()
            self.subplotReference.axis("off")
            self.subplotReference.set_title('Reference Voice')
        except Exception as e:
            raise NotImplementedError(f'Error: {e}\nHandle all other errors.')
            
        self.canvas.draw_idle()

    
    def update_recording_plot(self):
        try:
            y, sr = librosa.load(f'{self.recording_audio_path}/{self.word_index}.wav')
            self.subplotRecording.specgram(y,Fs=512)
        except FileNotFoundError:
            print(f'No {self.recording_audio_path} audio found') # no action is needed
            self.subplotRecording.clear()
            self.subplotRecording.axis("off")
            self.subplotRecording.set_title('Your Voice')
        except Exception as e:
            raise NotImplementedError(f'Error: {e}\nHandle all other errors.')

        self.canvas.draw_idle()

    @threading_decorator
    def _timeout(self,duration: int) -> None:
        sleep(duration)
        print("Streams NOT closed. TIMED OUT")
        exit()
        
    def close(self):
        self._timeout(5)
        self.stream_in.stop_stream()
        self.stream_in.close()
        self.stream_out.close()
        self.p.terminate()
        self.parent.quit()
        self.parent.destroy()
        print("Streams closed")
        exit()


if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(True, True)
    app = MainWindow(root)
    root.protocol('WM_DELETE_WINDOW', app.close)
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.close()