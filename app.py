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

    prompt = (
        f"Translate the following text from {source} to {target_name}. "
        f"Preserve the original meaning. Return only the translation:\n\n{text}"
    )

    if provider == "Local NLLB-200":
        tokenizer, model = load_local_model()
        tokenizer.src_lang = source_code
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),
                max_new_tokens=150,
            )

        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    if provider == "Gemini API":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Add GEMINI_API_KEY to your .env file.")

        client = genai.Client(
            api_key=api_key,
            http_options={"timeout": 15000},
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
        return response.text.strip()

    if provider == "OpenAI API":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Add OPENAI_API_KEY to your .env file.")

        client = OpenAI(
            api_key=api_key,
            timeout=15.0,
            max_retries=0,
        )
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        if not response.output_text:
            raise ValueError("OpenAI returned an empty response.")
        return response.output_text.strip()

    if provider == "Groq API":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Add GROQ_API_KEY to your .env file.")

        client = Groq(
            api_key=api_key,
            timeout=15.0,
            max_retries=0,
        )
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.choices[0].message.content
        if not result:
            raise ValueError("Groq returned an empty response.")
        return result.strip()

    raise ValueError("Unsupported model selected.")


def translate_with_timer(text, source, target, provider):
    if provider == "Local NLLB-200":
        return translate_text(text, source, target, provider)

    status = st.empty()
    start = time.perf_counter()
    future = executor.submit(translate_text, text, source, target, provider)

    try:
        while not future.done():
            elapsed = int(time.perf_counter() - start)
            status.info(f"Translating with {provider}... {elapsed}s")

            if elapsed >= 20:
                future.cancel()
                raise TimeoutError(f"{provider} exceeded the 20-second limit.")

            time.sleep(0.25)

        result = future.result()
        status.empty()
        return result
    except Exception:
        status.empty()
        raise

