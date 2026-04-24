import streamlit as st
import pandas as pd
import time
from transformers import pipeline

st.set_page_config(page_title="Translator MP", page_icon="🌍")
st.title('Translator MP')
st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=150)

st.markdown("""
### Instrukcja obsługi
Ta aplikacja służy do przetwarzania języka. 
* Wybierz **Wydźwięk emocjonalny tekstu**, aby sprawdzić, czy podany tekst ma wydźwięk pozytywny, czy negatywny.
* Wybierz **Tłumaczenie ENG -> GER**, aby przetłumaczyć angielski tekst na język niemiecki.
""")

st.success('Interfejs załadowany pomyślnie!')

st.header('Przetwarzanie języka naturalnego')

option = st.selectbox(
    "Wybierz opcję:",
    [
        "Wydźwięk emocjonalny tekstu",
        "Tłumaczenie ENG -> GER", 
    ],
)

if option == "Wydźwięk emocjonalny tekstu":
    text = st.text_area(label="Wpisz tekst po angielsku")
    if st.button("Analizuj"):
        if text:
            with st.spinner("Trwa analiza tekstu"):
                classifier = pipeline("sentiment-analysis")
                answer = classifier(text)
                st.success("Zakończono analizę!")
                st.write(answer)
        else:
            st.error("Proszę wpisać tekst do analizy!")

elif option == "Tłumaczenie ENG -> GER":
    text = st.text_area(label="Wpisz tekst po angielsku do przetłumaczenia")
    if st.button("Tłumacz"):
        if text:
            with st.spinner("Trwa tłumaczenie"):
                try:
                    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")
                    
                    answer = translator(text)
                    st.success("Tłumaczenie zakończone sukcesem!")
                    st.write("**Tłumaczenie:**", answer[0]['translation_text'])
                except Exception as e:
                    st.error(f"Wystąpił błąd: {e}")
        else:
            st.warning("Musisz wprowadzić tekst")

st.markdown("---")
st.write("Numer indeksu: s25402")