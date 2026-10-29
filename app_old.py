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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { color-scheme: dark; }
    .stApp { background: radial-gradient(ellipse at 15% 0%, #172554 0%, #0b1120 38%, #080d18 100%); color: #e5e7eb; }
    [data-testid="stHeader"] { background: rgba(8, 13, 24, 0.78); }
    [data-testid="stAppViewContainer"] .main .block-container { max-width: 1200px; padding-top: 2.2rem; padding-bottom: 3rem; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.035em; }
    h1 { color: #f8fafc; font-size: clamp(2rem, 4vw, 3rem) !important; }
    h2, h3 { color: #e2e8f0; }
    p, label, [data-testid="stCaptionContainer"] { color: #aab7cc; }
    .hero-card { padding: 1.65rem 1.8rem; border: 1px solid rgba(129, 140, 248, .25); border-radius: 22px; background: linear-gradient(120deg, rgba(30, 41, 79, .88), rgba(15, 23, 42, .82)); box-shadow: 0 18px 55px rgba(0,0,0,.22); margin-bottom: 1.5rem; }
    .eyebrow { color: #a5b4fc; text-transform: uppercase; font-size: .76rem; font-weight: 700; letter-spacing: .16em; margin-bottom: .65rem; }
    .hero-copy { color: #cbd5e1; font-size: 1.02rem; margin: .5rem 0 0 0; }
    .section-label { color: #a5b4fc; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .13em; margin-bottom: .3rem; }
    div[data-testid="stTextArea"] textarea, div[data-testid="stSelectbox"] > div > div { background: #111b2e !important; color: #f1f5f9 !important; border: 1px solid #293750 !important; border-radius: 12px !important; }
    div[data-testid="stTextArea"] textarea:focus { border-color: #818cf8 !important; box-shadow: 0 0 0 1px #818cf8 !important; }
    div.stButton > button, div.stDownloadButton > button { border-radius: 11px; border: 1px solid #4f46e5; font-weight: 700; min-height: 2.8rem; transition: all .18s ease; }
    div.stButton > button[kind="primary"] { background: linear-gradient(100deg, #6366f1, #8b5cf6); color: white; border: 0; }
    div.stButton > button:hover, div.stDownloadButton > button:hover { border-color: #a5b4fc; transform: translateY(-1px); }
    [data-testid="stMetric"] { background: linear-gradient(145deg, rgba(24, 36, 60, .95), rgba(15, 23, 42, .95)); border: 1px solid #293750; padding: 1rem 1.1rem; border-radius: 16px; }
    [data-testid="stMetricLabel"] { color: #a5b4fc !important; }
    [data-testid="stMetricValue"] { color: #f8fafc !important; }
    [data-testid="stExpander"] { border: 1px solid #293750; border-radius: 13px; background: rgba(15, 23, 42, .68); }
    [data-testid="stDataFrame"] { border: 1px solid #293750; border-radius: 12px; overflow: hidden; }
    [data-testid="stAlert"] { border-radius: 12px; }
    hr { border-color: #25324a; }
    @media (max-width: 700px) { [data-testid="stAppViewContainer"] .main .block-container { padding: 1.2rem 1rem 2rem; } .hero-card { padding: 1.25rem; } }
    </style>
    """,
    unsafe_allow_html=True,
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

    "Groq API",

    "Local NLLB-200",

    "Gemini API",

    "OpenAI API",

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

def reset_analysis(text, provider):

    st.session_state.analysis_active = True

    st.session_state.analysis_text = text

    st.session_state.active_provider = provider

    st.session_state.results = []

    st.session_state.next_index = 0

    st.session_state.initialized = False

    st.session_state.fallback_pending = False

    st.session_state.fallback_deadline = None

    st.session_state.fallback_error = ""

    st.session_state.failed_provider = ""

st.markdown(
    """
    <div class="hero-card">
      <div class="eyebrow">NLP LAB · MULTILINGUAL ANALYSIS</div>
      <h1>Indian Language Token Analysis</h1>
      <p class="hero-copy">Translate, inspect, and compare tokenization across English, Hindi, Bengali, and Tamil.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">01 / Input text</div>', unsafe_allow_html=True)
text = st.text_area(
    "Enter text in English, Hindi, Bengali, or Tamil",
    placeholder="Start with a sentence in any supported language…",
    height=140,
)

col1, col2 = st.columns([1.35, 1], gap="medium")
with col1:
    analyze = st.button("✦  Analyze text", type="primary", use_container_width=True)
with col2:
    provider = st.selectbox("Translation model", MODEL_OPTIONS)

if analyze:

    if not text.strip():

        st.warning("Please enter some text.")

        st.stop()

    reset_analysis(text.strip(), provider)

# Handle cloud API failure and fallback countdown.

if st.session_state.get("fallback_pending", False):

    remaining = math.ceil(st.session_state.fallback_deadline - time.time())

    st.error(

        f"{st.session_state.failed_provider} failed: "

        f"{st.session_state.fallback_error}"

    )

    st.warning(

        f"Switching to Local NLLB-200 in {max(0, remaining)} seconds..."

    )

    col1, col2 = st.columns(2)

    with col1:

        switch_now = st.button("Switch now", use_container_width=True)

    with col2:

        retry = st.button("Retry selected model", use_container_width=True)

    if switch_now:

        st.session_state.active_provider = "Local NLLB-200"

        st.session_state.fallback_pending = False

        st.rerun()

    if retry:

        st.session_state.active_provider = st.session_state.failed_provider

        st.session_state.fallback_pending = False

        st.rerun()

    if remaining <= 0:

        st.session_state.active_provider = "Local NLLB-200"

        st.session_state.fallback_pending = False

        st.rerun()

    time.sleep(1)

    st.rerun()

# Run analysis one language at a time.

if st.session_state.get("analysis_active", False):

    try:

        if not st.session_state.initialized:

            with st.spinner("Detecting language..."):

                tokenizer, detector = load_analysis_tools()

                detected = detector.detect_language_of(

                    st.session_state.analysis_text

                )

            if detected not in DETECTOR_LANGUAGES:

                st.error("Language not supported or could not be detected.")

                st.session_state.analysis_active = False

                st.stop()

            source = DETECTOR_LANGUAGES[detected]

            st.session_state.source_language = source

            st.session_state.initialized = True

            st.success(f"Detected language: {source}")

        tokenizer, _ = load_analysis_tools()

        languages = list(LANGUAGES.keys())

        index = st.session_state.next_index

        if index < len(languages):

            language = languages[index]

            active_provider = st.session_state.active_provider

            st.progress((index + 1) / len(languages), text=f"Processing language {index + 1} of {len(languages)}")

            try:

                start_translation = time.perf_counter()

                translated = translate_with_timer(

                    st.session_state.analysis_text,

                    st.session_state.source_language,

                    language,

                    active_provider,

                )

                translation_time = time.perf_counter() - start_translation

            except Exception as error:

                if active_provider != "Local NLLB-200":

                    st.session_state.failed_provider = active_provider

                    st.session_state.fallback_error = str(error)

                    st.session_state.fallback_deadline = time.time() + 7

                    st.session_state.fallback_pending = True

                    st.rerun()

                else:

                    raise

            start_tokenization = time.perf_counter()

            token_ids = tokenizer(

                translated,

                add_special_tokens=False,

            )["input_ids"]

            tokenization_time = time.perf_counter() - start_tokenization

            st.session_state.results.append({

                "Language": language,

                "Translated Text": translated,

                "Characters": len(translated),

                "Tokens": len(token_ids),

                "Tokenization Time (ms)": round(tokenization_time * 1000, 3),

                "Translation Time (s)": round(translation_time, 3),

                "Model Used": active_provider,

            })

            st.session_state.next_index += 1

            st.rerun()

        else:

            st.session_state.analysis_active = False

            st.success("Analysis completed!")

    except Exception as error:

        st.error(f"Analysis failed: {error}")

        st.session_state.analysis_active = False

# Display results after all translations are completed.

if (

    not st.session_state.get("analysis_active", False)

    and not st.session_state.get("fallback_pending", False)

    and st.session_state.get("results")

):

    df = pd.DataFrame(st.session_state.results)

    st.subheader("🌐 Translations")

    for _, row in df.iterrows():

        with st.expander(row["Language"], expanded=True):

            st.write(row["Translated Text"])

            st.caption(f"Model: {row['Model Used']}")

    st.subheader("📊 Token Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric("Languages", len(df))

    col2.metric("Total tokens", int(df["Tokens"].sum()))

    col3.metric(

        "Total translation time",

        f"{df['Translation Time (s)'].sum():.2f} s",

    )

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("📈 Token Count Comparison")

    st.bar_chart(df.set_index("Language")["Tokens"])

    st.download_button(

        "⬇  Download results as CSV",

        df.to_csv(index=False).encode("utf-8-sig"),

        file_name="language_token_analysis.csv",

        mime="text/csv",

    )

    st.caption(

        "Translation time includes API/network latency for cloud models. "

        "Tokenization uses MuRIL for every translation. API costs are not calculated."

    )
