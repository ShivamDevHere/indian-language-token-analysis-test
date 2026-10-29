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


st.title("Indian Language Token Analysis")
st.write("Translate and compare tokenization across four languages.")

text = st.text_area(
    "Enter text in English, Hindi, Bengali, or Tamil",
    placeholder="Type your sentence here...",
    height=120,
)

col1, col2 = st.columns([2, 1])
with col1:
    analyze = st.button("Analyze", type="primary", use_container_width=True)
with col2:
    provider = st.selectbox(
        "Translation model",
        MODEL_OPTIONS,
        label_visibility="collapsed",
    )

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
            st.write(f"Progress: {index + 1}/{len(languages)} languages")

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

    st.subheader("Translations")
    for _, row in df.iterrows():
        with st.expander(row["Language"], expanded=True):
            st.write(row["Translated Text"])
            st.caption(f"Model: {row['Model Used']}")

    st.subheader("Token Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Languages", len(df))
    col2.metric("Total tokens", int(df["Tokens"].sum()))
    col3.metric(
        "Total translation time",
        f"{df['Translation Time (s)'].sum():.2f} s",
    )

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.subheader("Token Count Comparison")
    st.bar_chart(df.set_index("Language")["Tokens"])

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name="language_token_analysis.csv",
        mime="text/csv",
    )

    st.caption(
        "Translation time includes API/network latency for cloud models. "
        "Tokenization uses MuRIL for every translation. API costs are not calculated."
    )
