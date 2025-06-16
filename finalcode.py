import os

from fontTools.misc.arrayTools import scaleRect

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tkinter as tk
from tkinter import scrolledtext, filedialog
import gradio as gr
import threading
import pygame
import time
import socket
import webview
import mat73
import numpy as np
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from functions.func_filters import butter_bandpass_filter
from functions import func_preproc as preproc
from functions import func_classifier as classifier
from kokoro import KPipeline
import torchaudio
import tempfile
import soundfile as sf
import webbrowser

# Fix Unicode output issues
def clean_unicode(text):
    return text.encode('ascii', errors='ignore').decode()

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pygame.mixer.init()

EEG = None
PREDICTION_RESULT = ""

def play_audio(audio_file):
    if audio_file is None:
        return "No audio file provided!"
    try:
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        return "Audio played successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

def get_available_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def train_model(EEG_data):
    baseline = [-200, 0]
    frame = [0, 600]
    for n_calib in range(len(EEG_data['train'])):
        cur_eeg = EEG_data['train'][n_calib]
        data = np.asarray(cur_eeg['data'])
        srate = cur_eeg['srate']
        data = butter_bandpass_filter(data, 0.5, 10, srate, 4)
        markers = cur_eeg['markers_target']
        targetID = np.where(markers == 1)[0]
        nontargetID = np.where(markers == 2)[0]
        tmp_targetEEG = preproc.extractEpoch3D(data, targetID, srate, baseline, frame, False)
        tmp_nontargetEEG = preproc.extractEpoch3D(data, nontargetID, srate, baseline, frame, False)
        if n_calib == 0:
            targetEEG = tmp_targetEEG
            nontargetEEG = tmp_nontargetEEG
        else:
            targetEEG = np.dstack((targetEEG, tmp_targetEEG))
            nontargetEEG = np.dstack((nontargetEEG, tmp_nontargetEEG))
    down_target = preproc.decimation_by_avg(targetEEG, 24)
    down_nontarget = preproc.decimation_by_avg(nontargetEEG, 24)
    feat_target = np.reshape(down_target, (down_target.shape[0] * down_target.shape[1], down_target.shape[2])).T
    feat_nontarget = np.reshape(down_nontarget, (down_nontarget.shape[0] * down_nontarget.shape[1], down_nontarget.shape[2])).T
    y_target = np.ones((feat_target.shape[0], 1))
    y_nontarget = -np.ones((feat_nontarget.shape[0], 1))
    feat_train = np.vstack((feat_target, feat_nontarget))
    y_train = np.vstack((y_target, y_nontarget))
    np.random.seed(101)
    idx_train = np.arange(feat_train.shape[0])
    np.random.shuffle(idx_train)
    feat_train = feat_train[idx_train, :]
    y_train = y_train[idx_train, :]
    feat_column = np.array(range(feat_train.shape[1]))
    feat_best_column, stats_best = classifier.stepwise_linear_model(feat_train, feat_column, y_train, 0.08)
    argsort_pval = np.argsort(stats_best.pvalues)
    feat_selec_column = feat_best_column[argsort_pval[range(60)]]
    feat_train_select = feat_train[:, feat_selec_column]
    scaler = StandardScaler()
#    feat_train_scaled = scaler.fit_transform(feat_train_select)
#    y_train_bin = (y_train + 1) // 2
#    model_nn = Sequential([
#        Dense(64, activation='relu', input_shape=(feat_train_scaled.shape[1],)),
#        Dropout(0.3),
#        Dense(32, activation='relu'),
#        Dropout(0.3),
#        Dense(1, activation='sigmoid')
#    ])
#    model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#    model_nn.fit(feat_train_scaled, y_train_bin, epochs=60, batch_size=32, validation_split = 0.3, verbose=1)
    model = LinearRegression()
    model.fit(feat_train_select, y_train)
    return model, feat_selec_column

def test_model(model, feat_selec_column, EEG_data):
    global PREDICTION_RESULT
    PREDICTION_RESULT = ""
    spellermatrix = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_")
    Config_P3speller = {"seq_code": range(1, 13), "full_repeat": 15, "spellermatrix": spellermatrix}
    for n_test in range(len(EEG_data['test'])):
        cur_eeg = EEG_data['test'][n_test]
        data = np.asarray(cur_eeg['data'])
        srate = cur_eeg['srate']
        data = butter_bandpass_filter(data, 0.5, 10, srate, 4)
        markers_seq = cur_eeg['markers_seq']
        letter_idx = np.where(np.isin(markers_seq, Config_P3speller['seq_code']))[0]
        unknownEEG = preproc.extractEpoch3D(data, letter_idx, srate, [-200, 0], [0, 600], False)
        down_unknown = preproc.decimation_by_avg(unknownEEG, 24)
        feat_unknown = np.reshape(down_unknown, (down_unknown.shape[0] * down_unknown.shape[1], down_unknown.shape[2])).T
        pred_unknown = model.predict(feat_unknown[:, feat_selec_column])
        ans_letters = preproc.detect_letter_P3speller(
            pred_unknown,
            int(cur_eeg['nbTrials'] / (len(Config_P3speller['seq_code']) * Config_P3speller['full_repeat'])),
            cur_eeg['text_to_spell'],
            letter_idx,
            markers_seq,
            Config_P3speller
        )
        result_str = f" {ans_letters['text_result']}"
        PREDICTION_RESULT += result_str + "\n"

def load_eeg_data(file_path):
    try:
        return mat73.loadmat(file_path)
    except Exception as e:
        print(f"Error loading .mat file: {str(e)}")
        return None

def generate_audio(text):
    try:
        pipeline = KPipeline(lang_code='a')
        zgenerator = pipeline(
            text,
            voice='jf_nezumi',
            speed=1,
            split_pattern=r'\n+'
        )
        audio_segments = []
        for i, (gs, ps, audio) in enumerate(zgenerator):
            print(i)
            print(clean_unicode(gs))  # Cleaned text output
            print(clean_unicode(ps))  # Cleaned phoneme output
            audio_segments.append(audio)
        if audio_segments:
            full_audio = np.concatenate(audio_segments)
            output_path = os.path.join(tempfile.gettempdir(), "kokoro_output.wav")
            sf.write(output_path, full_audio, 24000)
            return output_path
        else:
            return None
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return None

def launch_gradio_with_eeg(mat_file_path, on_result_ready_callback):
    port = get_available_port()
    data_dir = "C:/users/varki/PycharmProjects/data/"
    data = mat73.loadmat(data_dir+'s{:02d}.mat'.format(int(10)))
    model, feat_selec_column = train_model(data)
    EEG_data = load_eeg_data(mat_file_path)
    if EEG_data is None:
        processing_result = "Error loading EEG data"
    else:
        test_model(model, feat_selec_column, EEG_data)
        processing_result = PREDICTION_RESULT
    on_result_ready_callback(processing_result)

    with gr.Blocks(title="EEG and Speech Interface") as interface:
        gr.Markdown(f"### EEG Processing Result:\n```\n{processing_result}\n```")
        with gr.Row():
            with gr.Column():
                # Pre-populate the text input field with the result
                input_text = gr.Textbox(
                    label="EEG Processing Output", 
                    placeholder="Text from EEG prediction...", 
                    value=processing_result,  # Set the default value here
                    interactive=False  # Making the textbox read-only
                )
                submit_btn = gr.Button("Generate Speech")
            with gr.Column():
                audio_out = gr.Audio(label="Generated Speech", autoplay=True)

        submit_btn.click(
            fn=generate_audio,
            inputs=input_text,
            outputs=audio_out,
            api_name="generate_speech"
        )

    threading.Thread(
        target=interface.launch,
        kwargs={
            "server_name": "localhost",
            "server_port": port,
            "share": False,
            "inbrowser": False,
            "prevent_thread_lock": True
        },
        daemon=True
    ).start()

    return f"http://localhost:{port}"

def predict_from_eeg():
    global EEG, PREDICTION_RESULT
    mat_file_path = filedialog.askopenfilename(
        title="Select EEG Data File",
        filetypes=[("MATLAB files", "*.mat"), ("All files", "*.*")]
    )
    if not mat_file_path:
        text_window.insert(tk.END, "\nNo file selected for EEG processing.\n")
        return

    EEG = mat_file_path
    text_window.insert(tk.END, f"\nSelected EEG file: {EEG}\nProcessing...\n")

    def on_result_ready(result_text):
        text_window.insert(tk.END, f"\n{result_text}\n")
        text_window.see(tk.END)

    gradio_url = launch_gradio_with_eeg(EEG, on_result_ready)

    def wait_and_create_window():
        time.sleep(2)
        webview.create_window("EEG Data Processing & TTS", gradio_url, width=1000, height=700)
        app.after(100, webview.start)

    threading.Thread(target=wait_and_create_window, daemon=True).start()

# --------------------- TKINTER GUI ---------------------
import tkinter as tk
from tkinter import scrolledtext

def on_enter(e):
    e.widget['bg'] = '#c0392b'  # darker red on hover

def on_leave(e):
    e.widget['bg'] = '#e74c3c'  # original button color

app = tk.Tk()
app.title("Speech Generation for ALS Patients")
app.geometry("1600x1300")
app.configure(bg="#f0f8ff")

# Header
header_frame = tk.Frame(app, bg="#2c3e50", pady=20)
header_frame.pack(fill=tk.X, padx=10, pady=(10, 0))

title_label = tk.Label(
    header_frame,
    text="Speech Generation for ALS Patients",
    font=("Segoe UI", 26, "bold"),
    fg="white",
    bg="#2c3e50"
)
title_label.pack(pady=10)

# Content frame
content_frame = tk.Frame(app, bg="#f0f8ff")
content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Info frame with some padding and border radius illusion
info_frame = tk.Frame(content_frame, bg="#e6f2ff", bd=0, highlightthickness=1, highlightbackground="#b3c6ff")
info_frame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")
info_frame.grid_propagate(False)
info_frame.config(width=280, height=500)  # fixed size for neatness

info_label = tk.Label(
    info_frame,
    text="ALS Facts:\n\n"
         "• Progressive neurodegenerative disease\n"
         "• Affects nerve cells controlling muscles\n"
         "• Symptoms include difficulty speaking\n\n"
         "HELP\n",
    font=("Segoe UI", 13),
    bg="#e6f2ff",
    justify=tk.LEFT,
    padx=15,
    pady=15
)
info_label.pack(anchor="nw")

def open_link(url):
    webbrowser.open_new(url)

papers_text = tk.Text(
    info_frame,
    height=8,
    font=("Arial", 11),
    bg="#e6f2ff",
    relief=tk.FLAT,
    wrap=tk.WORD,
    cursor="arrow"
)
papers_text.pack(padx=10, pady=10)

papers_text.insert(tk.END, "ALS Association\n\n", "link1")
papers_text.insert(tk.END, "Asha Ek Hope Foundation\n\n", "link2")
papers_text.insert(tk.END, "Stem Cell Treatment\n\n", "link3")

# Configure tags for styling and click binding
papers_text.tag_config("link1", foreground="blue", underline=1)
papers_text.tag_bind("link1", "<Button-1>", lambda e: open_link("https://www.als.org/support"))

papers_text.tag_config("link2", foreground="blue", underline=1)
papers_text.tag_bind("link2", "<Button-1>", lambda e: open_link("https://www.als-mnd.org/directory/asha-ek-hope-foundation-for-mndals/"))

papers_text.tag_config("link3", foreground="blue", underline=1)
papers_text.tag_bind("link3", "<Button-1>", lambda e: open_link("https://www.stemcellcareindia.com/diseases/stem-cell-treatment-for-amyotrophic-lateral-sclerosis-india/"))

# Disable editing
papers_text.config(state=tk.DISABLED)


# Control frame
control_frame = tk.Frame(content_frame, bg="white", bd=2, relief=tk.RIDGE)
control_frame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")
control_frame.grid_propagate(False)
control_frame.config(width=560, height=500)

# Text window
text_window = scrolledtext.ScrolledText(
    control_frame,
    width=60,
    height=18,
    font=("Segoe UI", 12),
    wrap=tk.WORD,
    bg="white",
    fg="#333333",
    relief=tk.FLAT,
    bd=2,
)
text_window.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

# Button frame
button_frame = tk.Frame(control_frame, bg="white")
button_frame.pack(pady=10)

# Button with hover effect
eeg_button = tk.Button(
    button_frame,
    text="Predict from EEG",
    font=("Segoe UI", 14, "bold"),
    bg="#333333",
    fg="white",
    activebackground="#c0392b",
    activeforeground="white",
    relief=tk.FLAT,
    padx=20,
    pady=10,
    command=lambda: predict_from_eeg()
)
eeg_button.pack()

eeg_button.bind("<Enter>", on_enter)
eeg_button.bind("<Leave>", on_leave)

# Grid configuration for responsiveness
content_frame.grid_columnconfigure(0, weight=1, uniform="group1")
content_frame.grid_columnconfigure(1, weight=2, uniform="group1")
content_frame.grid_rowconfigure(0, weight=1)

app.mainloop()
