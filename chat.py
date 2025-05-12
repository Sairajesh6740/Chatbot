import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter import ttk
import sounddevice as sd
import numpy as np
from scipy.io import wavfile
import tempfile
import azure.cognitiveservices.speech as speechsdk
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
import os


class VoiceAssistant:
    """
    This class represents a voice assistant using Azure APIs.
    """

    def __init__(self, azure_speech_key, azure_region, azure_translator_key, azure_translator_region):
        # Azure API Key and Region for Speech and Translator
        self.azure_speech_key = azure_speech_key
        self.azure_region = azure_region
        self.azure_translator_key = azure_translator_key
        self.azure_translator_region = azure_translator_region

        # Translator Client Setup
        self.translator_client = TextTranslationClient(
            endpoint=f"https://{self.azure_translator_region}.api.cognitive.microsoft.com",
            credential=AzureKeyCredential(self.azure_translator_key)
        )

    def listen(self):
        """
        Records audio from the user and transcribes it using Azure Speech-to-Text.
        """
        try:
            duration = 5  # Record for 5 seconds
            fs = 46000  # Sample rate
            audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
            sd.wait()

            # Save the NumPy array to a temporary wav file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
                wavfile.write(temp_wav_file.name, fs, audio)
                temp_wav_path = temp_wav_file.name

            # Azure Speech-to-Text
            speech_config = speechsdk.SpeechConfig(subscription=self.azure_speech_key, region=self.azure_region)
            audio_input = speechsdk.AudioConfig(filename=temp_wav_path)
            recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcript = result.text
            else:
                transcript = "Sorry, I didn't catch that."
            
            return transcript
        except Exception as e:
            return f"Error: {str(e)}"

    def translate_text(self, text, target_language="en"):
        """
        Translates the input text to the specified language (default is English).
        """
        try:
            response = self.translator_client.translate(
                content=text,
                target_language=target_language
            )
            translated_text = response[0].translations[0].text
            return translated_text
        except Exception as e:
            return f"Error in translation: {str(e)}"

    def think(self, text):
        """
        Generates a response (dummy implementation).
        """
        # Here you can add logic to generate responses. For now, it's an echo bot.
        response = f"You said: {text}"
        return response

    def speak(self, text):
        """
        Converts text to speech and plays it using Azure Text-to-Speech.
        """
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.azure_speech_key, region=self.azure_region)
            synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
            result = synthesizer.speak_text_async(text).get()

            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                raise Exception(f"Error synthesizing speech: {result.reason}")
        except Exception as e:
            print(f"Error in speak: {e}")


class VoiceAssistantUI:
    """
    GUI for the Voice Assistant with styled UI.
    """

    def __init__(self, assistant):
        self.assistant = assistant
        self.root = tk.Tk()
        self.root.title("Voice Assistant")
        self.root.geometry("650x650")
        self.root.configure(bg="#2C3E50")  # Dark background with a sleek tint

        # Apply a modern theme to all ttk widgets
        style = ttk.Style()
        style.configure('TButton',
                        font=("Helvetica", 16, "bold"),
                        padding=15,
                        relief="flat",
                        background="#4B9CD3",
                        foreground="black",
                        width=20)
        style.map('TButton', background=[('active', '#2A7DAB')])  # Button color changes on hover
        style.configure('TLabel', font=("Helvetica", 16, "bold"), foreground="black")
        style.configure('TScrolledText', font=("Helvetica", 12), background="#34495E", foreground="black", wrap="word", height=12)
        style.configure('TFrame', background="#2C3E50")

        # Title Label with more styling
        title_label = ttk.Label(self.root, text="🎤 Voice Assistant 🤖", anchor="center", font=("Helvetica", 20, "bold"), foreground="black")
        title_label.pack(pady=30)

        # Welcome message
        welcome_label = ttk.Label(self.root, text="Welcome! 😊\nYour personal assistant is here to help.", anchor="center", font=("Helvetica", 16, "italic"), foreground="#1ABC9C")
        welcome_label.pack(pady=10)

        # Frame to hold buttons with padding
        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(pady=20)

        # Button to start listening
        self.listen_button = ttk.Button(button_frame, text="🎤 Start Listening", command=self.handle_listen)
        self.listen_button.grid(row=0, column=0, padx=20)

        # Button to end listening
        self.end_button = ttk.Button(button_frame, text="❌ End Listening", state=tk.DISABLED, command=self.handle_end_listen)
        self.end_button.grid(row=0, column=1, padx=20)

        # Scrollable text box for displaying conversation
        self.conversation_box = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=60, height=14, font=("Helvetica", 12), bg="#34495E", fg="white", bd=0, highlightthickness=0)
        self.conversation_box.pack(pady=10)
        self.conversation_box.insert(tk.END, "Assistant: How can I help you today?\n")
        
        # Customize scrollbars for aesthetics
        self.conversation_box.tag_configure("Me", foreground="lightpink")
        self.conversation_box.tag_configure("Alpha Assistant", foreground="lightblue")
        
    def handle_listen(self):
        """
        Handles the listening, thinking, and speaking workflow.
        """
        self.conversation_box.insert(tk.END, "Listening...\n")
        self.root.update()

        # Disable "Start Listening" button and enable "End Listening" button
        self.listen_button.config(state=tk.DISABLED)
        self.end_button.config(state=tk.NORMAL)

        # Listen and process user input
        user_input = self.assistant.listen()
        self.conversation_box.insert(tk.END, f"User: {user_input}\n", "user")
        self.root.update()

        if "goodbye" in user_input.strip().lower():
            response = "Goodbye! Have a great day! 😊"
            self.conversation_box.insert(tk.END, f"Assistant: {response}\n", "assistant")
            self.assistant.speak(response)
            messagebox.showinfo("Voice Assistant", "Goodbye! 😊")
            self.root.quit()
            return

        # Translate user input to English
        translated_input = self.assistant.translate_text(user_input)
        self.conversation_box.insert(tk.END, f"Translated: {translated_input}\n", "assistant")
        
        # Think and respond
        response = self.assistant.think(user_input)
        self.conversation_box.insert(tk.END, f"Assistant: {response}\n", "assistant")
        self.root.update()

        # Speak response
        self.assistant.speak(response)

    def handle_end_listen(self):
        """
        Ends the listening session and resets the buttons.
        """
        self.conversation_box.insert(tk.END, "Listening session ended.\n", "assistant")
        self.root.update()

        # Disable "End Listening" button and enable "Start Listening" button
        self.end_button.config(state=tk.DISABLED)
        self.listen_button.config(state=tk.NORMAL)

        # Thank you message at the end
        thank_you_label = ttk.Label(self.root, text="Thank you for using the Voice Assistant! 🤗", anchor="center", font=("Helvetica", 16, "italic"), foreground="#F39C12")
        thank_you_label.pack(pady=10)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    # Replace with your Azure Speech resource key, region, and Translator credentials
    azure_speech_key = "YOUR_AZURE_SPEECH_KEY"  # Replace with your key
    azure_region = "YOUR_AZURE_REGION"  # Replace with your region
    azure_translator_key = "YOUR_AZURE_TRANSLATOR_KEY"  # Replace with your Translator key
    azure_translator_region = "YOUR_AZURE_TRANSLATOR_REGION"  # Replace with your Translator region

    assistant = VoiceAssistant(azure_speech_key, azure_region, azure_translator_key, azure_translator_region)
    ui = VoiceAssistantUI(assistant)
    ui.run()
