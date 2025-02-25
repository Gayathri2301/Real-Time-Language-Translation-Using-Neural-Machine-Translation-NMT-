import speech_recognition as sr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pyttsx3

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")
        return None
    except sr.RequestError:
        print("Could not request results, check your internet connection.")
        return None

def translate_text(text, src_lang="en_XX", tgt_lang="fr_XX"):
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    
    tokenizer.src_lang = src_lang
    encoded_text = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang])
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(f"Translated: {translated_text}")
    return translated_text

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def main():
    print("Real-Time Language Translator")
    spoken_text = recognize_speech()
    if spoken_text:
        translated_text = translate_text(spoken_text, "en_XX", "fr_XX")  # English to French
        speak_text(translated_text)

if __name__ == "__main__":
    main()
