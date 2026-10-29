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

