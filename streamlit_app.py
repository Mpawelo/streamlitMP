import streamlit as st
import pandas as pd
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="Translator MP", page_icon="🌍")
st.title('Translator MP')
st.image("https://pja.edu.pl/wp-content/uploads/2024/07/PJATK_Warszawa_Budynek_A_4-1.jpg", width=150)

st.markdown("""
### Instrukcja obsługi
Ta aplikacja służy do przetwarzania języka. 
* Wybierz **Wydźwięk emocjonalny tekstu**, aby sprawdzić, czy podany tekst ma wydźwięk pozytywny, czy negatywny.
* Wybierz **Tłumaczenie ENG -> GER**, aby przetłumaczyć angielski tekst na język niemiecki.
""")

st.success('Interfejs załadowany pomyślnie!')


option = st.selectbox(
    "Wybierz opcję:",
    [
        "Wydźwięk emocjonalny tekstu",
        "Tłumaczenie ENG -> GER", 
    ],
)

if option == "Wydźwięk emocjonalny tekstu":
    st.header('Przetwarzanie języka naturalnego')
    text = st.text_area(label="Wpisz tekst po angielsku")
    if st.button("Analizuj"):
        if text:
            with st.spinner("Trwa analiza tekstu"):
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)
                st.success("Zakończono analizę!")
                st.balloons()
                st.write(answer)
        else:
            st.error("Proszę wpisać tekst do analizy!")

elif option == "Tłumaczenie ENG -> GER":
    st.header('Tłumacz')
    text = st.text_area(label="Wpisz tekst po angielsku do przetłumaczenia")
    if st.button("Tłumacz"):
        if text:
            with st.spinner("Trwa tłumaczenie"):
                try:
                    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
                    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")
                    
                    inputs = tokenizer(text, return_tensors="pt")
                    outputs = model.generate(**inputs)
                    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    st.success("Tłumaczenie zakończone sukcesem!")
                    st.write("**Tłumaczenie:**", translated_text)
                    st.balloons()
                except Exception as e:
                    st.error(f"Wystąpił błąd: {e}")
        else:
            st.warning("Musisz wprowadzić tekst")

st.markdown("---")
st.write("Numer indeksu: s25402")