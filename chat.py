"""
Enterprise-Grade Voice Assistant with Azure Integration
Author: NARRA SAI RAJESH REDDY
Version: 2.1
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
import threading
import queue
import configparser
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv

import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import AzureError

# Configure logging
logging.basicConfig(
    filename='assistant.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# Load environment variables
load_dotenv()

@dataclass
class AzureConfig:
    """Configuration container for Azure services"""
    speech_key: str
    speech_region: str
    translator_key: str
    translator_region: str
    translator_endpoint: str = "https://api.cognitive.microsofttranslator.com"

class VoiceAssistant:
    """Core assistant functionality using Azure Cognitive Services"""
    
    def __init__(self, config: AzureConfig):
        self.config = config
        self._validate_config()
        self._init_clients()
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.sample_rate = 48000  # Professional audio quality

    def _validate_config(self):
        """Ensure valid configuration"""
        if not all([
            self.config.speech_key,
            self.config.speech_region,
            self.config.translator_key,
            self.config.translator_region
        ]):
            raise ValueError("Invalid Azure configuration")

    def _init_clients(self):
        """Initialize Azure service clients"""
        try:
            # Speech config with neural voice
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.config.speech_key,
                region=self.config.speech_region
            )
            self.speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
            
            # Translator client
            self.translator_client = TextTranslationClient(
                endpoint=self.config.translator_endpoint,
                credential=AzureKeyCredential(self.config.translator_key),
                region=self.config.translator_region
            )
        except AzureError as e:
            logging.critical(f"Azure service initialization failed: {str(e)}")
            raise

    def listen_continuous(self, callback: callable):
        """Start continuous listening with real-time processing"""
        def audio_callback(indata, frames, time, status):
            if status:
                logging.warning(f"Audio stream status: {status}")
            self.audio_queue.put(indata.copy())

        self.is_listening = True
        self.audio_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            dtype=np.int16
        )
        self.audio_stream.start()

        def process_audio():
            while self.is_listening:
                try:
                    audio_data = self.audio_queue.get(timeout=1)
                    self._process_audio_chunk(audio_data, callback)
                except queue.Empty:
                    continue

        self.processing_thread = threading.Thread(target=process_audio)
        self.processing_thread.start()

    def _process_audio_chunk(self, audio_data: np.ndarray, callback: callable):
        """Process audio chunk through Azure Speech-to-Text"""
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            wavfile.write(temp_file.name, self.sample_rate, audio_data)
            
            audio_config = speechsdk.audio.AudioConfig(filename=temp_file.name)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            result = recognizer.recognize_once_async().get()
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                callback(result.text)

    def stop_listening(self):
        """Stop audio processing"""
        self.is_listening = False
        if self.audio_stream:
            self.audio_stream.stop()
        if self.processing_thread:
            self.processing_thread.join()

    def translate_text(self, text: str, target_lang: str) -> Tuple[Optional[str], Optional[str]]:
        """Translate text using Azure Translator"""
        try:
            response = self.translator_client.translate(
                content=[text],
                target_language=target_lang
            )
            translation = response[0].translations[0]
            return translation.text, translation.to
        except AzureError as e:
            logging.error(f"Translation failed: {str(e)}")
            return None, str(e)

    def synthesize_speech(self, text: str):
        """Convert text to speech using Azure Neural TTS"""
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            error_msg = f"Speech synthesis failed: {result.reason}"
            logging.error(error_msg)
            raise RuntimeError(error_msg)

class AssistantGUI:
    """Professional GUI implementation"""
    
    def __init__(self, assistant: VoiceAssistant):
        self.assistant = assistant
        self.root = tk.Tk()
        self.root.title("Enterprise Voice Assistant")
        self.root.geometry("800x600")
        self._configure_styles()
        self._create_widgets()
        self.conversation_history = []

    def _configure_styles(self):
        """Setup modern UI styling"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TButton', 
                      font=('Segoe UI', 12), 
                      padding=10,
                      foreground='#2c3e50',
                      background='#3498db')
        
        style.configure('TLabel', 
                       font=('Segoe UI', 12), 
                       background='#f0f0f0')
        
        style.configure('TCombobox', font=('Segoe UI', 12))
        self.root.configure(bg='#f0f0f0')

    def _create_widgets(self):
        """Create UI components"""
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(pady=20)
        
        ttk.Label(header_frame, 
                text="🎤 Enterprise Voice Assistant", 
                font=('Segoe UI', 18, 'bold')).grid(row=0, column=0)
        
        # Language controls
        lang_frame = ttk.Frame(self.root)
        lang_frame.pack(pady=10)
        
        ttk.Label(lang_frame, text="Input Language:").grid(row=0, column=0)
        self.input_lang = ttk.Combobox(lang_frame, 
                                      values=['en-US', 'es-ES', 'fr-FR', 'de-DE'])
        self.input_lang.set('en-US')
        self.input_lang.grid(row=0, column=1, padx=10)
        
        ttk.Label(lang_frame, text="Output Language:").grid(row=0, column=2)
        self.output_lang = ttk.Combobox(lang_frame, 
                                       values=['en', 'es', 'fr', 'de'])
        self.output_lang.set('en')
        self.output_lang.grid(row=0, column=3, padx=10)
        
        # Conversation history
        self.conversation = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=('Segoe UI', 12),
            bg='white',
            padx=15,
            pady=15,
            state='normal'
        )
        self.conversation.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Control buttons
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        self.listen_btn = ttk.Button(
            btn_frame,
            text="Start Listening",
            command=self.toggle_listening
        )
        self.listen_btn.grid(row=0, column=0, padx=10)
        
        ttk.Button(
            btn_frame,
            text="Exit",
            command=self.root.quit
        ).grid(row=0, column=1, padx=10)
        
        # Status bar
        self.status = ttk.Label(self.root, 
                              text="Ready", 
                              relief=tk.SUNKEN,
                              anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_listening(self):
        """Toggle listening state"""
        if not hasattr(self, 'is_listening') or not self.is_listening:
            self.start_listening()
        else:
            self.stop_listening()

    def start_listening(self):
        """Start audio capture"""
        self.is_listening = True
        self.listen_btn.config(text="Stop Listening")
        self.status.config(text="Listening...")
        self.assistant.listen_continuous(self.process_transcript)

    def stop_listening(self):
        """Stop audio capture"""
        self.is_listening = False
        self.listen_btn.config(text="Start Listening")
        self.status.config(text="Ready")
        self.assistant.stop_listening()

    def process_transcript(self, transcript: str):
        """Handle recognized speech"""
        self.conversation.config(state='normal')
        self.conversation.insert(tk.END, f"User: {transcript}\n")
        
        try:
            translated, target_lang = self.assistant.translate_text(
                transcript, 
                self.output_lang.get()
            )
            
            response = f"Processed in {target_lang}: {translated}"
            
            self.assistant.synthesize_speech(response)
            self.conversation.insert(tk.END, f"Assistant: {response}\n")
            
        except Exception as e:
            logging.error(f"Processing error: {str(e)}")
            self.conversation.insert(tk.END, f"System Error: {str(e)}\n")
            
        self.conversation.see(tk.END)
        self.conversation.config(state='disabled')

    def run(self):
        """Start application"""
        self.root.mainloop()

def load_config() -> AzureConfig:
    """Load Azure configuration from environment"""
    return AzureConfig(
        speech_key=os.getenv("AZURE_SPEECH_KEY"),
        speech_region=os.getenv("AZURE_SPEECH_REGION"),
        translator_key=os.getenv("AZURE_TRANSLATOR_KEY"),
        translator_region=os.getenv("AZURE_TRANSLATOR_REGION")
    )

if __name__ == "__main__":
    try:
        config = load_config()
        assistant = VoiceAssistant(config)
        gui = AssistantGUI(assistant)
        gui.run()
    except Exception as e:
        logging.critical(f"Application failed: {str(e)}")
        messagebox.showerror("Fatal Error", 
            "Application failed to initialize. Check logs for details.")
