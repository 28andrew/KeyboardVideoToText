import tkinter as tk
from tkinter import simpledialog, messagebox
import cv2
import os
import time
import json
import random
import string
from collections import deque
from PIL import Image, ImageTk
import sounddevice as sd
import numpy as np
import threading
import re
import queue
import av
import subprocess
import tempfile

# Constants
FPS = 30  # Target frame rate
PREVIEW_FPS = 15  # Lower frame rate for preview
PREVIEW_WIDTH, PREVIEW_HEIGHT = 800, 450
MAX_CHARS = 100
DATA_DIR = 'data'
RECORD_DIR = 'recordings'
ALLOWED_CHARS = set(string.ascii_letters + string.digits + string.punctuation + ' ')


class TypingDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Typing Data Collector")
        self.root.geometry(f"{PREVIEW_WIDTH + 350}x{PREVIEW_HEIGHT + 200}")
        # Create directories if they don't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RECORD_DIR, exist_ok=True)

        # Video capture and audio stream
        self.cap = None
        self.audio_stream = None
        # Buffers - only store during active typing
        self.frame_buffer = []
        self.frame_timestamps = []
        self.audio_buffer = []
        self.audio_timestamps = []
        # State flags
        self.is_recording = False
        self.record_start_time = None
        self.last_keystroke_time = None
        self.keystrokes = []
        self.state = 'press_enter'
        self.popup = None
        self.finished_typing = False
        self.preview_visible = True  # New flag to track preview visibility

        # Threading related
        self.frame_lock = threading.Lock()
        self.preview_thread_running = True
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame

        # Setup
        try:
            self._select_camera()
            self._select_microphone()
            self._start_audio_stream()
            self._load_sentences()
            self._setup_ui()

            # Start the threaded frame capture
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()

            # Start the preview update
            self._update_preview()

            # Set up clean shutdown
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        except Exception as e:
            messagebox.showerror('Error', f'Setup error: {str(e)}')
            self.root.destroy()

    def _on_close(self):
        """Ensure clean shutdown of threads"""
        self.preview_thread_running = False

        # Clean up resources
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except:
                pass

        self.root.destroy()

    def _select_camera(self):
        cams = []
        for i in range(10):  # Check more camera indices
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ok, _ = cap.read()
                    cap.release()
                    if ok:
                        cams.append(i)
            except:
                pass

        if not cams:
            messagebox.showerror('Error', 'No webcams found. Please connect a webcam and try again.')
            self.root.destroy()
            return

        choice = simpledialog.askinteger('Select Webcam', f'Available cameras: {cams}\nEnter index:')
        if choice not in cams:
            messagebox.showerror('Error', 'Invalid index.')
            self.root.destroy()
            return

        self.cap = cv2.VideoCapture(choice)
        if not self.cap.isOpened():
            # Try again with a different backend
            self.cap = cv2.VideoCapture(choice, cv2.CAP_AVFOUNDATION)  # macOS backend

        # Set properties only if the camera opened successfully
        if self.cap.isOpened():
            # Set precise properties
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set standard resolution
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # Get actual frame size for consistent encoding
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Check that we got a valid camera
            if self.frame_width == 0 or self.frame_height == 0:
                messagebox.showerror('Error', 'Failed to initialize camera properly')
                self.root.destroy()
        else:
            messagebox.showerror('Error', f'Failed to open camera with index {choice}')
            self.root.destroy()

    def _select_microphone(self):
        try:
            devs = sd.query_devices()
            ins = [(i, d['name']) for i, d in enumerate(devs) if d['max_input_channels'] > 0]
            if not ins:
                messagebox.showerror('Error', 'No audio devices.')
                self.root.destroy()
                return

            prompt = 'Select Mic:\n' + '\n'.join(f"{i}: {n}" for i, n in ins)
            choice = simpledialog.askinteger('Select Microphone', prompt)
            if choice not in [i for i, _ in ins]:
                messagebox.showerror('Error', 'Invalid mic.')
                self.root.destroy()
                return

            self.audio_device = choice
            self.samplerate = int(sd.query_devices(choice)['default_samplerate'])
        except Exception as e:
            messagebox.showerror('Error', f'Microphone setup error: {str(e)}')
            self.root.destroy()

    def _start_audio_stream(self):
        try:
            self.audio_stream = sd.InputStream(
                device=self.audio_device,
                channels=1,
                samplerate=self.samplerate,
                callback=self._audio_callback
            )
            self.audio_stream.start()
        except Exception as e:
            messagebox.showerror('Error', f'Audio stream error: {str(e)}')
            self.root.destroy()

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")

        # Only record during active typing
        if self.is_recording and not self.finished_typing:
            now = time.time()
            self.audio_buffer.append(indata.copy())
            self.audio_timestamps.append(now)

    def _capture_frames(self):
        """Dedicated thread for frame capture at maximum possible rate"""
        last_frame_time = 0
        while self.preview_thread_running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.01)  # Short sleep if camera not available
                continue

            try:
                ret, frame = self.cap.read()
                if ret:
                    now = time.time()

                    # Calculate and print actual frame rate
                    if last_frame_time > 0:
                        fps = 1.0 / (now - last_frame_time)
                        # if len(self.frame_buffer) % 30 == 0:  # Print every 30 frames
                        #     print(f"Actual capture rate: {fps:.2f} fps")
                    last_frame_time = now

                    # Update latest frame for preview (overwrite any existing one)
                    try:
                        # Put without blocking, replace any existing item
                        if self.frame_queue.full():
                            self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait((now, frame))
                    except queue.Full:
                        pass  # Skip this frame for preview

                    # Store frame for recording if needed
                    if self.is_recording and not self.finished_typing:
                        with self.frame_lock:
                            self.frame_buffer.append(frame.copy())
                            self.frame_timestamps.append(now)
            except Exception as e:
                print(f"Error capturing frame: {e}")
                time.sleep(0.01)  # Short sleep on error

    def _update_preview(self):
        """Update the preview at a reasonable rate without slowing down capture"""
        try:
            # Only update the preview if it's visible
            if self.preview_visible:
                # Get the latest frame from the queue if available
                try:
                    _, frame = self.frame_queue.get_nowait()

                    # Create preview image
                    disp = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
                    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
                    self.preview_label.imgtk = img
                    self.preview_label.config(image=img)
                except queue.Empty:
                    pass  # No new frame available
        except Exception as e:
            print(f"Error updating preview: {e}")

        # Schedule next update at preview frame rate
        if self.preview_thread_running:
            self.root.after(int(1000 / PREVIEW_FPS), self._update_preview)

    def _load_sentences(self):
        self.file_sentences = []
        self.weights = []

        # Check if data directory exists
        if not os.path.exists(DATA_DIR):
            messagebox.showerror('Error', f'Data directory {DATA_DIR} does not exist')
            self.root.destroy()
            return

        # Load sentence files
        for f in os.listdir(DATA_DIR):
            if f.endswith('.txt'):
                try:
                    with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as file:
                        lines = [l.strip() for l in file if l.strip()]
                        valid = [l for l in lines if all(c in ALLOWED_CHARS for c in l) and len(l) <= MAX_CHARS]
                        if valid:
                            self.file_sentences.append(valid)
                            self.weights.append(len(valid))
                except Exception as e:
                    print(f"Error loading file {f}: {e}")

        if not self.file_sentences:
            messagebox.showerror('Error', 'No valid sentences found in data directory')
            self.root.destroy()

    def _toggle_preview(self):
        """Toggle the visibility of the preview frame"""
        if self.preview_visible:
            # Hide preview
            self.preview_frame.pack_forget()
            self.toggle_button.config(text="Show Preview")
            self.preview_visible = False

            # Resize the window to be more compact when preview is hidden
            current_width = self.root.winfo_width()
            new_height = self.root.winfo_height() - PREVIEW_HEIGHT - 20  # Subtract preview height + padding
            self.root.geometry(f"{current_width}x{new_height}")
        else:
            # Show preview
            self.preview_frame.pack(before=self.controls_frame, padx=10, pady=10)
            self.toggle_button.config(text="Hide Preview")
            self.preview_visible = True

            # Resize window back to original size
            current_width = self.root.winfo_width()
            new_height = self.root.winfo_height() + PREVIEW_HEIGHT + 20  # Add preview height + padding
            self.root.geometry(f"{current_width}x{new_height}")

    def _setup_ui(self):
        self.root.title('Press Enter to start')

        # Controls Frame (will contain the toggle button)
        self.controls_frame = tk.Frame(self.root)
        self.controls_frame.pack(fill='x', padx=10, pady=5)

        # Toggle button for preview
        self.toggle_button = tk.Button(
            self.controls_frame,
            text="Hide Preview",
            command=self._toggle_preview,
            width=15,
            font=('Helvetica', 12)
        )
        self.toggle_button.pack(side='left')

        # Preview Frame
        self.preview_frame = tk.Frame(self.root)
        self.preview_frame.pack(padx=10, pady=10)
        self.preview_label = tk.Label(self.preview_frame)
        self.preview_label.pack()

        # Instruction
        self.instruction_frame = tk.Frame(self.root)
        self.instruction_frame.pack(fill='x', padx=10)
        self.instruction_label = tk.Label(
            self.instruction_frame,
            text='Press Enter to start',
            wraplength=PREVIEW_WIDTH + 300,
            font=('Helvetica', 20)
        )
        self.instruction_label.pack(fill='x', pady=10)

        # Entry
        self.entry_frame = tk.Frame(self.root)
        self.entry_frame.pack(padx=10, pady=5)
        self.entry = tk.Entry(
            self.entry_frame,
            width=100,
            font=('Helvetica', 18)
        )
        self.entry.pack(pady=5, fill='x', expand=True)

        # Disallow backspace
        self.entry.bind('<KeyPress-BackSpace>', lambda e: 'break')

        # Keystrokes & text check
        self.entry.bind('<KeyPress>', self._on_key_event, add='+')
        self.entry.bind('<KeyRelease>', self._check_text, add='+')

        # Enter to start
        self.entry.bind('<Return>', self._handle_enter)
        self.root.bind_all('<Return>', self._handle_enter)

        self.entry.focus_set()

    def _on_key_event(self, event):
        if self.state != 'typing' or self.finished_typing:
            return

        now = time.time()
        ev = 'down' if event.type == tk.EventType.KeyPress else 'up'

        # Start recording on first keystroke
        if not self.is_recording and ev == 'down':
            self.is_recording = True
            self.record_start_time = now
            with self.frame_lock:
                self.frame_buffer = []
                self.frame_timestamps = []
            self.audio_buffer = []
            self.audio_timestamps = []

        self.last_keystroke_time = now

        # Record keystroke with time offset from first keystroke
        if self.record_start_time:
            rel_time = now - self.record_start_time
            self.keystrokes.append({'event': ev, 'key': event.keysym, 'time': now, 'rel_time': rel_time})

    def _handle_enter(self, event):
        if self.state == 'press_enter':
            self._on_start()
        return 'break'

    def _sample_sentence(self):
        tot = sum(self.weights)
        r = random.uniform(0, tot)
        u = 0
        for lines, w in zip(self.file_sentences, self.weights):
            u += w
            if r <= u:
                return random.choice(lines)
        # Fallback in case something goes wrong
        return "Sample sentence."

    def _on_start(self):
        if self.state != 'press_enter':
            return

        self.state = 'typing'
        raw = self._sample_sentence().replace('\n', ' ')
        raw = re.sub(r' {2,}', ' ', raw)
        self.target_text = raw.strip()
        self.instruction_label.config(text=self.target_text)
        self.entry.delete(0, tk.END)
        self.entry.focus_set()

        # Reset all recording state
        self.is_recording = False
        self.finished_typing = False
        self.record_start_time = None
        self.last_keystroke_time = None
        self.keystrokes = []
        with self.frame_lock:
            self.frame_buffer = []
            self.frame_timestamps = []
        self.audio_buffer = []
        self.audio_timestamps = []
        self.root.title('Typing...')

    def _check_text(self, event=None):
        if self.state != 'typing':
            return

        # collapse spaces
        txt = self.entry.get()
        typed = txt.lstrip()

        if typed and not self.target_text.startswith(typed):
            messagebox.showinfo('Retry', 'Incorrect typing — restart')
            self._reset()
            return

        if typed == self.target_text:
            # We're done typing, set the flag to stop recording
            self.finished_typing = True
            self.entry.unbind('<KeyRelease>')

            self.popup = tk.Toplevel(self.root)
            self.popup.title('Saving')
            self.popup.transient(self.root)
            self.popup.grab_set()
            self.popup.protocol('WM_DELETE_WINDOW', lambda: None)
            tk.Label(self.popup, text='Saving, please wait...', font=('Helvetica', 16)).pack(padx=20, pady=20)
            threading.Thread(target=self._finish, args=(True,), daemon=True).start()

    def _create_video_with_pyav(self, frames, timestamps, output_file, target_fps=30):
        """Create a video file with precise frame timing using PyAV"""
        if not frames or not timestamps:
            raise ValueError("No frames to encode")

        # Calculate duration
        duration = timestamps[-1] - timestamps[0]
        print(f"Video duration: {duration:.2f} seconds")

        # Create output container
        container = av.open(output_file, mode='w')

        # Add video stream
        stream = container.add_stream('h264', rate=target_fps)
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        stream.pix_fmt = 'yuv420p'
        stream.time_base = av.Fraction(1, 1000)  # Use milliseconds as time base

        # Scale timestamps to start from zero
        rel_timestamps = [t - timestamps[0] for t in timestamps]

        # Calculate exact number of frames needed
        total_frames = int(duration * target_fps) + 1

        # Create frame sequence with proper timing
        for i in range(total_frames):
            target_time = i / target_fps

            # Find closest frame by timestamp
            closest_idx = min(range(len(rel_timestamps)),
                              key=lambda j: abs(rel_timestamps[j] - target_time))

            # Convert to PyAV frame
            frame = frames[closest_idx]
            av_frame = av.VideoFrame.from_ndarray(frame, format='bgr24')

            # Set exact timestamp in milliseconds
            pts_ms = int(i * 1000 / target_fps)
            av_frame.pts = pts_ms

            # Encode and mux
            for packet in stream.encode(av_frame):
                container.mux(packet)

        # Flush encoder
        for packet in stream.encode(None):
            container.mux(packet)

        # Close container
        container.close()

        return output_file, duration

    def _finish(self, success):
        base = time.strftime('%Y%m%d_%H%M%S')

        if not success:
            self.root.after(0, lambda: [
                self.popup.destroy(),
                messagebox.showinfo('Retry', 'Incorrect typing')])
            self.root.after(0, self._reset)
            return

        try:
            # Make sure we have valid recording
            if not self.record_start_time or not self.last_keystroke_time:
                raise ValueError("Missing recording timing information")

            # Create recordings directory if it doesn't exist
            os.makedirs(RECORD_DIR, exist_ok=True)

            # Calculate the true duration based on keystrokes
            true_duration = self.last_keystroke_time - self.record_start_time
            print(f"True typing duration: {true_duration:.2f} seconds")

            # Process frames
            with self.frame_lock:
                # Only use frames between first keystroke and last keystroke
                valid_frames = []
                valid_timestamps = []

                for frame, timestamp in zip(self.frame_buffer, self.frame_timestamps):
                    if self.record_start_time <= timestamp <= self.last_keystroke_time + 0.05:
                        valid_frames.append(frame)
                        valid_timestamps.append(timestamp)

            if not valid_frames:
                raise ValueError("No valid video frames captured during typing")

            # Process audio
            valid_audio = []
            valid_audio_timestamps = []

            for audio_data, timestamp in zip(self.audio_buffer, self.audio_timestamps):
                if self.record_start_time <= timestamp <= self.last_keystroke_time + 0.05:
                    valid_audio.append(audio_data)
                    valid_audio_timestamps.append(timestamp)

            if not valid_audio:
                raise ValueError("No valid audio data captured during typing")

            # Output info about captured media
            print(f"Valid frames: {len(valid_frames)}")
            print(f"Valid audio chunks: {len(valid_audio)}")

            # Calculate actual frame rate
            if len(valid_timestamps) > 1:
                time_diffs = [valid_timestamps[i + 1] - valid_timestamps[i]
                              for i in range(len(valid_timestamps) - 1)]
                avg_frame_interval = sum(time_diffs) / len(time_diffs)
                actual_fps = 1.0 / avg_frame_interval
                print(f"Actual frame rate: {actual_fps:.2f} fps")
            else:
                actual_fps = 30.0

            # Target a standard frame rate for output
            target_fps = 30.0

            # Create video file with PyAV
            vid_temp = os.path.join(RECORD_DIR, f'{base}_temp.mp4')
            try:
                self._create_video_with_pyav(valid_frames, valid_timestamps, vid_temp, target_fps)
            except Exception as e:
                print(f"PyAV error: {e}")
                # Fallback to OpenCV if PyAV fails
                print("Falling back to OpenCV for video encoding")
                h, w = valid_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(vid_temp, fourcc, target_fps, (w, h))

                if not out.isOpened():
                    raise ValueError(f"Failed to create video writer")

                for frame in valid_frames:
                    out.write(frame)
                out.release()

            # Create audio file
            audio_data = np.concatenate(valid_audio, axis=0)
            audio_duration = audio_data.shape[0] / self.samplerate
            print(f"Audio duration: {audio_duration:.2f} seconds")

            wav_temp = os.path.join(RECORD_DIR, f'{base}_temp.wav')
            import soundfile as sf
            sf.write(wav_temp, audio_data, self.samplerate)

            # Combine audio and video
            final = os.path.join(RECORD_DIR, f'{base}.mp4')

            # Use FFmpeg to combine with precise timing control
            cmd = [
                'ffmpeg', '-y',
                '-i', vid_temp,
                '-i', wav_temp,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-profile:v', 'main',
                '-pix_fmt', 'yuv420p',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v:0',
                '-map', '1:a:0',
                '-shortest',
                '-t', str(true_duration),  # Force exact duration
                '-async', '1',  # Keep audio sync
                '-movflags', '+faststart',
                final
            ]

            print("Executing ffmpeg...")
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print("FFmpeg output:")
                print(result.stdout)
                print(result.stderr)
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg error: {e}")
                print("FFmpeg stderr:")
                print(e.stderr)
                raise

            # Verify the output file exists
            if not os.path.exists(final) or os.path.getsize(final) == 0:
                raise ValueError(f"Failed to create output file {final}")

            # Clean up temporary files
            try:
                os.remove(vid_temp)
                os.remove(wav_temp)
            except Exception as e:
                print(f"Warning: Could not remove temp files: {e}")

            # Create JSON with keystroke data
            ks_data = [
                {
                    'event': k['event'],
                    'key': k['key'],
                    'timestamp_ms': int((k['time'] - self.record_start_time) * 1000)
                }
                for k in self.keystrokes
            ]

            # Verify final video duration
            try:
                probe_cmd = [
                    'ffprobe', '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    final
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                final_duration = float(result.stdout.strip())
                print(f"Final video duration: {final_duration:.2f} seconds")

                # Check for duration mismatch
                if abs(final_duration - true_duration) > 0.1:
                    print(f"WARNING: Video duration mismatch: expected {true_duration:.2f}, got {final_duration:.2f}")
            except Exception as e:
                print(f"Warning: Could not verify final duration: {e}")
                final_duration = true_duration

            # Save metadata
            data = {
                'text': self.target_text,
                'duration_sec': final_duration,
                'keystrokes': ks_data,
                'media': os.path.basename(final),
                'actual_fps': actual_fps
            }

            with open(os.path.join(RECORD_DIR, f'{base}.json'), 'w', encoding='utf-8') as jf:
                json.dump(data, jf, indent=2)

            self.root.after(0, self._on_saved, base)

        except Exception as e:
            error_msg = f"Error saving recording: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()

            if self.popup:
                self.root.after(0, lambda: [
                    self.popup.destroy(),
                    messagebox.showerror('Error', error_msg)])
            else:
                self.root.after(0, lambda: messagebox.showerror('Error', error_msg))

            self.root.after(0, self._reset)

    def _on_saved(self, base):
        for w in self.popup.winfo_children():
            w.destroy()

        tk.Label(self.popup, text=f'Saved {base}.mp4', font=('Helvetica', 16)).pack(padx=20, pady=10)
        tk.Button(self.popup, text="OK", command=self.popup.destroy).pack(pady=10)
        self.popup.after(200, lambda: self.popup.destroy() if self.popup else None)
        self._reset()

    def _reset(self):
        self.state = 'press_enter'
        self.finished_typing = False
        self.instruction_label.config(text='Press Enter to start')
        self.entry.bind('<KeyRelease>', self._check_text)
        self.entry.bind('<Return>', self._handle_enter)
        self.entry.delete(0, tk.END)
        # Refocus window and entry
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.entry.focus_set()
        self.root.title('Press Enter to Start')

    def __del__(self):
        # Clean up resources
        self.preview_thread_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        if hasattr(self, 'audio_stream') and self.audio_stream:
            try:
                self.audio_stream.stop()
                self.audio_stream.close()
            except:
                pass


if __name__ == '__main__':
    try:
        # Check dependencies
        try:
            import av

            print("Using PyAV for video processing")
        except ImportError:
            print("PyAV not found. Please install it with:")
            print("pip install av")
            print("Continuing with OpenCV fallback...")

        import soundfile as sf

        # Create necessary directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(RECORD_DIR, exist_ok=True)

        root = tk.Tk()
        app = TypingDataCollector(root)
        root.mainloop()
    except Exception as e:
        import traceback

        traceback.print_exc()
        messagebox.showerror('Fatal Error', f"Application failed to start: {str(e)}")