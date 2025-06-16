import tkinter as tk
from tkinter import scrolledtext, filedialog
import gradio as gr
import threading
import pygame
import time
import socket
import webview
import scipy.io
import joblib
import mat73
import numpy as np
from sklearn.linear_model import LinearRegression
from functions.func_filters import butter_bandpass_filter
from functions import func_preproc as preproc
from functions import func_classifier as classifier
from kokoro import KPipeline
import os
import torchaudio
import tempfile
import soundfile as sf

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
    mdl_linear = LinearRegression()
    mdl_linear.fit(feat_train_select, y_train)
    return mdl_linear, feat_selec_column

def test_model(mdl_linear, feat_selec_column, EEG_data):
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
        pred_unknown = mdl_linear.predict(feat_unknown[:, feat_selec_column])
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
    EEG_data = load_eeg_data(mat_file_path)
    if EEG_data is None:
        processing_result = "Error loading EEG data"
    else:
        mdl_linear, feat_selec_column = train_model(EEG_data)
        test_model(mdl_linear, feat_selec_column, EEG_data)
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
app = tk.Tk()
app.title("Speech Generation for ALS Patients")
app.geometry("900x700")
app.configure(bg="#f0f8ff")

header_frame = tk.Frame(app, bg="#2c3e50")
header_frame.pack(fill=tk.X, padx=10, pady=10)

title_label = tk.Label(
    header_frame,
    text="Speech Generation for ALS Patients",
    font=("Arial", 24, "bold"),
    fg="white",
    bg="#2c3e50"
)
title_label.pack(pady=15)

content_frame = tk.Frame(app, bg="#f0f8ff")
content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

info_frame = tk.Frame(content_frame, bg="#e6f2ff", bd=2, relief=tk.RIDGE)
info_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

info_label = tk.Label(
    info_frame,
    text="ALS Facts:\n\n"
         "• Progressive neurodegenerative disease\n"
         "• Affects nerve cells controlling muscles\n"
         "• Symptoms include difficulty speaking",
    font=("Arial", 12),
    bg="#e6f2ff",
    justify=tk.LEFT,
    padx=10,
    pady=10
)
info_label.pack()

papers_label = tk.Label(
    info_frame,
    text="\nHelpful Research Papers:\n\n"
         "1. ALS and Genetics\n"
         "2. ALS and Environmental Factors\n"
         "3. Recent Advances in ALS Research",
    font=("Arial", 11),
    bg="#e6f2ff",
    justify=tk.LEFT,
    padx=10,
    pady=10
)
papers_label.pack()

control_frame = tk.Frame(content_frame, bg="#ffffff", bd=2, relief=tk.RIDGE)
control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

text_window = scrolledtext.ScrolledText(
    control_frame,
    width=50,
    height=15,
    font=("Arial", 11),
    wrap=tk.WORD,
    bg="white",
    fg="#333333"
)
text_window.pack(padx=10, pady=10)

button_frame = tk.Frame(control_frame, bg="#ffffff")
button_frame.pack(pady=10)

eeg_button = tk.Button(
    button_frame,
    text="Predict from EEG",
    font=("Arial", 12, "bold"),
    bg="#e74c3c",
    fg="white",
    activebackground="#c0392b",
    activeforeground="white",
    relief=tk.FLAT,
    command=predict_from_eeg
)
eeg_button.pack()

content_frame.grid_columnconfigure(0, weight=1)
content_frame.grid_columnconfigure(1, weight=2)

app.mainloop()
