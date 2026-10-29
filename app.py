import os
import time
import math
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import pandas as pd
import torch
from dotenv import load_dotenv
from lingua import Language, LanguageDetectorBuilder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from google import genai
from openai import OpenAI
from groq import Groq

load_dotenv()

st.set_page_config(
    page_title="Indian Language Token Analysis",
    page_icon="🔤",
    layout="wide",
)

LANGUAGES = {
    "English": ("eng_Latn", "English"),
    "Hindi": ("hin_Deva", "Hindi"),
    "Bengali": ("ben_Beng", "Bengali"),
    "Tamil": ("tam_Taml", "Tamil"),
}

DETECTOR_LANGUAGES = {
    Language.ENGLISH: "English",
    Language.HINDI: "Hindi",
    Language.BENGALI: "Bengali",
    Language.TAMIL: "Tamil",
}

MODEL_OPTIONS = [
    "Local NLLB-200",
    "Gemini API",
    "OpenAI API",
    "Groq API",
]

executor = ThreadPoolExecutor(max_workers=2)


@st.cache_resource
def load_analysis_tools():
    tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    detector = LanguageDetectorBuilder.from_languages(
        *DETECTOR_LANGUAGES.keys()
    ).build()
    return tokenizer, detector


@st.cache_resource
def load_local_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


def translate_text(text, source, target, provider):
    if source == target:
        return text

    source_code = LANGUAGES[source][0]
    target_code, target_name = LANGUAGES[target]


