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
import matplotlib.figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np
import librosa
import pyaudio
import wave

from typing import Callable, List

# def file_handler_decorator(function):
#         def wrapper(*args, **kwargs):
#             try:
#                 function(*args, **kwargs)
#             except FileNotFoundError:
#                 print("No reference found") # no action is needed
#             except Exception as e:
#                 raise NotImplementedError(f'Error: {e}\nHandle all other errors.')
#         return wrapper

def threading_decorator(function: Callable[..., None]) -> Callable[..., None]:
    """
    Utility function to spawn processes on a separate thread.
    
    Returns
    -------
    Callable[..., None]
        The decorated function.
    """
    def wrapper(*args, **kwargs) -> None:
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

    words: List[str]
    word_index: int = 0
    reference_audio_path: str = 'Ryder'
    recording_audio_path: str = 'Artin'

    def __init__(self, parent: tk.Tk, *args, **kwargs):
        """
        Load words, create audio streams, and initialize UI.
        """
        
        super().__init__(parent, *args, **kwargs)

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

        self.parent.bind("<Button-1>", self.clip_specs)
        
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

        next_buttonn = ttk.Button(bot_frame, text = 'Next',command = self.next_word)
        next_buttonn.pack(side = tk.RIGHT, padx = 10, pady = 10)
        prev_button = ttk.Button(bot_frame, text = 'Previous', command = self.prev_word)
        prev_button.pack(side = tk.LEFT, padx = 10, pady = 10)
        
        self.record_var = tk.StringVar()
        self.record_var.set('Record')
        replayReferenceButton = ttk.Button(mid_frame, text = 'Play Reference', command = self.play_reference)
        replayReferenceButton.pack(side = tk.LEFT, padx = 10, pady = 10)
        self.recordButton = ttk.Button(mid_frame, textvariable = self.record_var, command = self.record)
        self.recordButton.pack(side = tk.LEFT, padx = 10, pady = 10)
        replayOwnButton = ttk.Button(mid_frame, text = 'Play Your Voice', command = self.play_recording)
        replayOwnButton.pack(side = tk.LEFT, padx = 10, pady = 10)
        # playBothButton = ttk.Button(mid_frame, text = 'Play Both')
        # playBothButton.pack(side = tk.LEFT, padx = 10, pady = 10)

        # fig = matplotlib.pyplot.specgram()
        fig = matplotlib.figure.Figure(figsize=(4, 3), dpi = 100) # type: ignore
       
        self.subplotReference = fig.add_subplot(121)
        self.subplotRecording = fig.add_subplot(122)
        self.subplotReference.axis('off')
        self.subplotRecording.axis('off')
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
    
    def clip_specs(self, event):
        x, y = event.x,event.y
        print(x,y)
        if x > 470 and x<770:
            try:
                data = []
                file_name = f'{self.recording_audio_path}/{self.word_index}.wav'
                with wave.open(file_name) as wf:
                    # new_data = []
                    data = wf.readframes(wf.getnframes())
                    data = data[int(wf.getnframes()*((x-470)/300))*2:]
                    # for i in range(0,44100,44100//1024):
                    #     if i > int(44100*(x-470/300)): 
                    #         new_data.append(wf.readframes(1024))
                
                open(file_name,'w').close()

                with wave.open(file_name,'wb') as wf:
                    wf.setframerate(44100)
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 2 Bytes per sample (pyaudio.paInt16)
                    wf.writeframes(data)
                
                self.play_recording()
                self.update_recording_plot()
                # self.writeFile(,data)
            except FileNotFoundError:
                print(f'Can not find SM audio file') # no action is needed
            except Exception as e:
                raise NotImplementedError(f'Error: {e}\nHandle all other errors.')
        print(x,y)
        
    def next_word(self) -> None:
        """
        Changes the current word to the next one. Updates the UI.        
        """
        
        self.word_index += 1
        self.text_var.set(self.words[self.word_index])
        self.update_reference_plot()
        self.update_recording_plot()

    def prev_word(self) -> None:
        """
        Changes the current word to the previous one. Updates the UI. 
        """
        
        self.word_index -= 1
        self.text_var.set(self.words[self.word_index])
        self.update_reference_plot()
        self.update_recording_plot()
    
    @staticmethod
    def loadText(file_name: str) -> List[str]:
        """
        Load words from words file in the form of a list of strings.
        
        Returns
        -------
        List[str]
            The list of each word in the file.
        """
        
        with open(file_name, 'r') as f: 
            return f.readlines()

    # @file_handler_decorator
    @threading_decorator
    def play_file(self, file_name: str) -> None:
        """
        Plays the requested audio file.
        
        Intended for (.wav) files.
        
        Raises
        ------
        NotImplementedError
            If any error other than a FileNotFoundError is raised.
        """
        
        try:
            with wave.open(file_name) as wf:
                #print(wf.getnchannels(),wf.getframerate(),self.p.get_format_from_width(wf.getsampwidth()))
                # data = wf.readframes(1024)
                
                # print(len(data))
                # while len(data)>=1024:
                #     self.stream_out.write(data,1024)
                #     data = wf.readframes(1024)
                self.stream_out.write(wf.readframes(wf.getnframes()),wf.getnframes())
        except FileNotFoundError:
            print(f'Can not find {file_name} audio file') # no action is needed
        except Exception as e:
            raise NotImplementedError(f'Error: {e}\nHandle all other errors.')
    

    def play_reference(self) -> None:
        """
        Wrapper function for playing the reference audio file.
        """
        
        self.play_file(f'{self.reference_audio_path}/{self.word_index}.wav')

    def play_recording(self) -> None:
        """
        Wrapper function for playing the recording audio file.
        """
        
        self.play_file(f'{self.recording_audio_path}/{self.word_index}.wav')
            
    @threading_decorator        
    def record(self) -> None:
        """
        Starts recording (one second).
        
        Updates the UI, handles file IO, and plays back the recording.
        """
        
        self.record_var.set('Recording')
        self.recordButton.config(command = lambda: None)
        self.stream_in.start_stream()
        
        recording_frames = (self.stream_in.read(1024) for _ in range(44100 // (1024 * 1)))

        self.writeFile(f'{self.recording_audio_path}/{self.word_index}.wav',recording_frames)

        self.stream_in.stop_stream()
        self.record_var.set("Record")
        self.recordButton.config(command = self.record)

   
    def writeFile(self,file_name,frames_data):

        open(file_name,'w').close()

        with wave.open(file_name,'wb') as wf:
            wf.setframerate(44100)
            wf.setnchannels(1)
            wf.setsampwidth(2) # 2 Bytes per sample (pyaudio.paInt16)
            wf.writeframes(b''.join(frames_data))
        
        self.play_recording()
        self.update_recording_plot()


    def update_reference_plot(self) -> None:
        """
        Update the reference plot (UI).
        
        Raises
        ------
        NotImplementedError
            If any error other than a FileNotFoundError is raised.
        """
        self.subplotReference.clear()
        self.subplotReference.axis("off")
        self.subplotReference.set_title('Reference Voice')
        try:
            y, _ = librosa.load(f'{self.reference_audio_path}/{self.word_index}.wav')
            self.subplotReference.specgram(y, Fs=512)
        except FileNotFoundError:
            print(f'No {self.reference_audio_path} audio found') # no action is needed
        except Exception as e:
            raise NotImplementedError(f'Error: {e}\nHandle all other errors.')
            
        self.canvas.draw_idle()

    
    def update_recording_plot(self) -> None:
        """
        Update the recording plot (UI).
        
        Raises
        ------
        NotImplementedError
            If any error other than a FileNotFoundError is raised.
        """
        self.subplotRecording.clear()
        self.subplotRecording.axis("off")
        self.subplotRecording.set_title('Your Voice')
        try:
            y, _ = librosa.load(f'{self.recording_audio_path}/{self.word_index}.wav')
            self.subplotRecording.specgram(y, Fs=512)
        except FileNotFoundError:
            print(f'No {self.recording_audio_path} audio found') # no action is needed
            
        except Exception as e:
            raise NotImplementedError(f'Error: {e}\nHandle all other errors.')

        self.canvas.draw_idle()

    @threading_decorator
    def _timeout(self, duration: int) -> None:
        """
        Helper function to ensure that the program exits even if cleanup fails.
        
        *Intended for internal use only*
        """
        
        sleep(duration)
        print('Streams NOT closed. TIMED OUT')
        exit()
        
    def close(self):
        """
        Exit program.
        """
        
        self._timeout(5)
        self.stream_in.stop_stream()
        self.stream_in.close()
        self.stream_out.close()
        self.p.terminate()
        self.parent.quit()
        self.parent.destroy()
        print('Streams closed')
        
        exit()


if __name__ == '__main__':
    root = tk.Tk()
    root.resizable(False, False)
    app = MainWindow(root)
    root.geometry("900x550")
    root.protocol('WM_DELETE_WINDOW', app.close)
    
    try:
        app.mainloop()
    except KeyboardInterrupt:
        app.close()