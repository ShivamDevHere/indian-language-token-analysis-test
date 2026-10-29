import os,time,math
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import pandas as pd
import torch
from dotenv import load_dotenv
from lingua import Language,LanguageDetectorBuilder
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from google import genai
from openai import OpenAI
from groq import Groq

load_dotenv()
st.set_page_config(page_title="Indian Language Token Analysis",page_icon="🔤",layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at 15% 0%,#172554,#0b1120 45%,#080d18);color:#e5e7eb}
[data-testid="stHeader"]{background:rgba(8,13,24,.8)}
[data-testid="stAppViewContainer"] .main .block-container{max-width:1200px;padding-top:1.5rem;padding-bottom:2rem}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.03em}
h1{color:#f8fafc;font-size:clamp(2rem,4vw,3rem)!important}
h2,h3{color:#e2e8f0}
p,label,[data-testid="stCaptionContainer"]{color:#aab7cc}
.hero{padding:1.5rem;border:1px solid #35416b;border-radius:18px;background:linear-gradient(120deg,#1e294f,#0f172a);margin-bottom:1.2rem}
.eyebrow,.section{color:#a5b4fc;font-size:.77rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase}
div[data-testid="stTextArea"] textarea,div[data-testid="stSelectbox"]>div>div{background:#111b2e!important;color:#f1f5f9!important;border:1px solid #293750!important;border-radius:10px!important}
div.stButton>button,div.stDownloadButton>button{border-radius:10px;font-weight:700;min-height:2.6rem}
div.stButton>button[kind="primary"]{background:linear-gradient(100deg,#6366f1,#8b5cf6);color:white;border:0}
[data-testid="stMetric"]{background:linear-gradient(145deg,#18243c,#0f172a);border:1px solid #293750;padding:.8rem;border-radius:13px}
[data-testid="stMetricLabel"]{color:#a5b4fc!important}
[data-testid="stMetricValue"]{color:#f8fafc!important}
[data-testid="stExpander"]{border:1px solid #293750;border-radius:11px;background:#0f172ab0}
hr{border-color:#25324a}
@media(max-width:700px){[data-testid="stAppViewContainer"] .main .block-container{padding:1rem}.hero{padding:1rem}}
</style>
""",unsafe_allow_html=True)

LANGUAGES={
    "English":("eng_Latn","English"),
    "Hindi":("hin_Deva","Hindi"),
    "Bengali":("ben_Beng","Bengali"),
    "Tamil":("tam_Taml","Tamil")
}
DETECTOR_LANGUAGES={
    Language.ENGLISH:"English",
    Language.HINDI:"Hindi",
    Language.BENGALI:"Bengali",
    Language.TAMIL:"Tamil"
}
MODEL_OPTIONS=["Groq API","Local NLLB-200","Gemini API","OpenAI API"]
USD_TO_INR=88.0
MODEL_PRICING={
    "Gemini API":{"input":0.30,"output":2.50,"model":"gemini-2.5-flash"},
    "OpenAI API":{"input":0.40,"output":1.60,"model":"gpt-4.1-mini"},
    "Groq API":{"input":0.15,"output":0.60,"model":"openai/gpt-oss-120b"}
}
executor=ThreadPoolExecutor(max_workers=2)

# Basic keyword screening only; not a safety classifier.
SAFETY_TERMS={
    "English":["kill yourself","how to make a bomb","i will kill you"],
    "Hindi":["खुद को मार","बम बनाना","मैं तुम्हें मार"],
    "Bengali":["নিজেকে মেরে","বোমা বানানো","আমি তোমাকে মারব"],
    "Tamil":["உன்னைக் கொல்ல","வெடிகுண்டு தயாரிக்க","உன்னை கொல்வேன்"]
}

def safety_check(text,language):
    lowered=text.casefold()
    matches=[term for term in SAFETY_TERMS.get(language,[]) if term.casefold() in lowered]
    return "Keyword match" if matches else "No keyword match",", ".join(matches) if matches else "None"

def estimate_api_cost(provider,input_tokens,output_tokens):
    rates=MODEL_PRICING.get(provider)
    if not rates:return 0.0
    return ((input_tokens*rates["input"]+output_tokens*rates["output"])/1_000_000)*USD_TO_INR

@st.cache_resource
def load_analysis_tools():
    tokenizer=AutoTokenizer.from_pretrained("google/muril-base-cased")
    detector=LanguageDetectorBuilder.from_languages(*DETECTOR_LANGUAGES.keys()).build()
    return tokenizer,detector

@st.cache_resource
def load_local_model():
    name="facebook/nllb-200-distilled-600M"
    return AutoTokenizer.from_pretrained(name),AutoModelForSeq2SeqLM.from_pretrained(name)

def translate_text(text,source,target,provider):
    if source==target:return text,0,0
    source_code=LANGUAGES[source][0]
    target_code,target_name=LANGUAGES[target]
    prompt=f"Translate the following text from {source} to {target_name}. Preserve its meaning and tone. Return only the translation:\n\n{text}"

    if provider=="Local NLLB-200":
        tokenizer,model=load_local_model()
        tokenizer.src_lang=source_code
        inputs=tokenizer(text,return_tensors="pt")
        with torch.no_grad():
            output=model.generate(**inputs,forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code),max_new_tokens=150)
        return tokenizer.batch_decode(output,skip_special_tokens=True)[0],0,0

    if provider=="Gemini API":
        key=os.getenv("GEMINI_API_KEY")
        if not key:raise ValueError("Missing GEMINI_API_KEY.")
        client=genai.Client(api_key=key,http_options={"timeout":15000})
        response=client.models.generate_content(model=MODEL_PRICING[provider]["model"],contents=prompt)
        if not response.text:raise ValueError("Gemini returned an empty response.")
        usage=getattr(response,"usage_metadata",None)
        return response.text.strip(),getattr(usage,"prompt_token_count",0) or 0,getattr(usage,"candidates_token_count",0) or 0

    if provider=="OpenAI API":
        key=os.getenv("OPENAI_API_KEY")
        if not key:raise ValueError("Missing OPENAI_API_KEY.")
        client=OpenAI(api_key=key,timeout=15,max_retries=0)
        response=client.responses.create(model=MODEL_PRICING[provider]["model"],input=prompt)
        if not response.output_text:raise ValueError("OpenAI returned an empty response.")
        usage=getattr(response,"usage",None)
        return response.output_text.strip(),getattr(usage,"input_tokens",0) or 0,getattr(usage,"output_tokens",0) or 0

    if provider=="Groq API":
        key=os.getenv("GROQ_API_KEY")
        if not key:raise ValueError("Missing GROQ_API_KEY.")
        client=Groq(api_key=key,timeout=15,max_retries=0)
        response=client.chat.completions.create(model=MODEL_PRICING[provider]["model"],messages=[{"role":"user","content":prompt}])
        result=response.choices[0].message.content
        if not result:raise ValueError("Groq returned an empty response.")
        usage=getattr(response,"usage",None)
        return result.strip(),getattr(usage,"prompt_tokens",0) or 0,getattr(usage,"completion_tokens",0) or 0

    raise ValueError("Unsupported model.")

def translate_with_timer(text,source,target,provider):
    if provider=="Local NLLB-200" or source==target:
        return translate_text(text,source,target,provider)
    status=st.empty()
    start=time.perf_counter()
    future=executor.submit(translate_text,text,source,target,provider)
    try:
        while not future.done():
            elapsed=int(time.perf_counter()-start)
            status.info(f"Translating with {provider}... {elapsed}s")
            if elapsed>=20:
                future.cancel()
                raise TimeoutError(f"{provider} exceeded the 20-second limit.")
            time.sleep(.25)
        result=future.result()
        status.empty()
        return result
    except Exception:
        status.empty()
        raise

def reset_analysis(text,provider):
    st.session_state.analysis_active=True
    st.session_state.analysis_text=text
    st.session_state.active_provider=provider
    st.session_state.results=[]
    st.session_state.next_index=0
    st.session_state.initialized=False
    st.session_state.input_analysis={}
    st.session_state.fallback_pending=False
    st.session_state.fallback_deadline=None
    st.session_state.fallback_error=""
    st.session_state.failed_provider=""

st.markdown("""
<div class="hero">
<div class="eyebrow">NLP LAB · MULTILINGUAL EVALUATION</div>
<h1>Indian Language Token Analysis</h1>
<p>Compare token efficiency, translation speed, estimated API costs and basic keyword screening across English, Hindi, Bengali and Tamil.</p>
</div>
""",unsafe_allow_html=True)

st.markdown('<div class="section">01 / Input text</div>',unsafe_allow_html=True)
text=st.text_area("Enter text in a supported language",placeholder="Enter English, Hindi, Bengali or Tamil text...",height=120)
col1,col2=st.columns([1.35,1])
with col1:
    analyze=st.button("✦ Analyze text",type="primary",use_container_width=True)
with col2:
    provider=st.selectbox("Translation model",MODEL_OPTIONS)

if analyze:
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()
    reset_analysis(text.strip(),provider)

if st.session_state.get("fallback_pending",False):
    remaining=math.ceil(st.session_state.fallback_deadline-time.time())
    st.error(f"{st.session_state.failed_provider} failed: {st.session_state.fallback_error}")
    st.warning(f"Switching to Local NLLB-200 in {max(0,remaining)} seconds...")
    c1,c2=st.columns(2)
    with c1:
        switch_now=st.button("Switch now",use_container_width=True)
    with c2:
        retry=st.button("Retry selected model",use_container_width=True)
    if switch_now:
        st.session_state.active_provider="Local NLLB-200"
        st.session_state.fallback_pending=False
        st.rerun()
    if retry:
        st.session_state.active_provider=st.session_state.failed_provider
        st.session_state.fallback_pending=False
        st.rerun()
    if remaining<=0:
        st.session_state.active_provider="Local NLLB-200"
        st.session_state.fallback_pending=False
        st.rerun()
    time.sleep(1)
    st.rerun()

if st.session_state.get("analysis_active",False):
    try:
        tokenizer,detector=load_analysis_tools()
        if not st.session_state.initialized:
            with st.spinner("Detecting language and analyzing input..."):
                original=st.session_state.analysis_text
                detected=detector.detect_language_of(original)
                if detected not in DETECTOR_LANGUAGES:
                    st.error("Language not supported or could not be detected.")
                    st.session_state.analysis_active=False
                    st.stop()
                source=DETECTOR_LANGUAGES[detected]
                original_tokens=len(tokenizer(original,add_special_tokens=False)["input_ids"])
                safety_status,safety_matches=safety_check(original,source)
                st.session_state.source_language=source
                st.session_state.input_analysis={
                    "Language":source,
                    "Characters":len(original),
                    "Words":len(original.split()),
                    "MuRIL Tokens":original_tokens,
                    "Safety Screening":safety_status,
                    "Matched Keywords":safety_matches
                }
                st.session_state.initialized=True
            st.success(f"Detected language: {source}")

        languages=list(LANGUAGES.keys())
        index=st.session_state.next_index
        if index<len(languages):
            language=languages[index]
            active_provider=st.session_state.active_provider
            st.progress((index+1)/len(languages),text=f"Processing language {index+1} of {len(languages)}")
            try:
                start=time.perf_counter()
                translated,input_tokens,output_tokens=translate_with_timer(
                    st.session_state.analysis_text,
                    st.session_state.source_language,
                    language,
                    active_provider
                )
                translation_time=time.perf_counter()-start
            except Exception as error:
                if active_provider!="Local NLLB-200":
                    st.session_state.failed_provider=active_provider
                    st.session_state.fallback_error=str(error)
                    st.session_state.fallback_deadline=time.time()+7
                    st.session_state.fallback_pending=True
                    st.rerun()
                raise

            start=time.perf_counter()
            token_ids=tokenizer(translated,add_special_tokens=False)["input_ids"]
            tokenization_time=(time.perf_counter()-start)*1000
            safety_status,safety_matches=safety_check(translated,language)
            st.session_state.results.append({
                "Language":language,
                "Translated Text":translated,
                "Characters":len(translated),
                "Words":len(translated.split()),
                "MuRIL Tokens":len(token_ids),
                "Tokenization Time (ms)":round(tokenization_time,3),
                "Translation Time (s)":round(translation_time,3),
                "Model Used":active_provider,
                "Input API Tokens":input_tokens,
                "Output API Tokens":output_tokens,
                "Estimated Cost (INR)":estimate_api_cost(active_provider,input_tokens,output_tokens),
                "Safety Screening":safety_status,
                "Matched Keywords":safety_matches
            })
            st.session_state.next_index+=1
            st.rerun()
        else:
            st.session_state.analysis_active=False
            st.success("Analysis completed!")
    except Exception as error:
        st.error(f"Analysis failed: {error}")
        st.session_state.analysis_active=False

if not st.session_state.get("analysis_active",False) and not st.session_state.get("fallback_pending",False) and st.session_state.get("results"):
    df=pd.DataFrame(st.session_state.results)
    original=st.session_state.input_analysis

    st.subheader("📝 Original Input Analysis")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Detected Language",original["Language"])
    c2.metric("Characters",original["Characters"])
    c3.metric("Words",original["Words"])
    c4.metric("MuRIL Tokens",original["MuRIL Tokens"])
    st.caption(f"Original text keyword screening: {original['Safety Screening']}. Matches: {original['Matched Keywords']}")

    st.subheader("🌐 Translations")
    for _,row in df.iterrows():
        with st.expander(row["Language"],expanded=True):
            st.write(row["Translated Text"])
            st.caption(f"Model: {row['Model Used']} · {row['Characters']} characters · {row['MuRIL Tokens']} MuRIL tokens")

    st.subheader("📊 Token Efficiency")
    c1,c2,c3=st.columns(3)
    c1.metric("Languages compared",len(df))
    c2.metric("Total translated tokens",int(df["MuRIL Tokens"].sum()))
    c3.metric("Total translation time",f"{df['Translation Time (s)'].sum():.2f} s")
    token_df=df[["Language","Characters","Words","MuRIL Tokens"]].copy()
    st.dataframe(token_df,use_container_width=True,hide_index=True,column_config={
        "MuRIL Tokens":st.column_config.ProgressColumn(
            "MuRIL Tokens",min_value=0,max_value=max(1,int(token_df["MuRIL Tokens"].max())),format="%d"
        )
    })
    st.caption("Token counts use the same MuRIL tokenizer for every language. Compare equivalent meanings; different translations can change token counts.")

    st.subheader("💰 Estimated API Cost")
    total_cost=float(df["Estimated Cost (INR)"].sum())
    c1,c2=st.columns(2)
    c1.metric("Total estimated API cost",f"₹{total_cost:.6f}")
    c2.metric("USD/INR conversion",f"₹{USD_TO_INR:.2f}")
    cost_df=df[["Language","Model Used","Input API Tokens","Output API Tokens","Estimated Cost (INR)"]].copy()
    st.dataframe(cost_df,use_container_width=True,hide_index=True,column_config={
        "Estimated Cost (INR)":st.column_config.NumberColumn("Estimated Cost (₹)",format="₹%.6f")
    })
    st.caption("Cost uses configured per-million-token rates and a fixed exchange-rate estimate. Actual provider pricing and charges may differ. Same-language results and local NLLB-200 have ₹0 API cost.")

    st.subheader("⚡ Performance")
    st.dataframe(df[["Language","Translation Time (s)","Tokenization Time (ms)"]],use_container_width=True,hide_index=True)

    st.subheader("🛡️ Basic Safety Screening")
    st.dataframe(df[["Language","Safety Screening","Matched Keywords"]],use_container_width=True,hide_index=True)
    st.warning("This is a simple keyword demonstration, not an AI safety classifier. No keyword match does not mean the text is safe. Add a validated multilingual safety model and labeled test data for real safety evaluation.")

    with st.expander("Full results"):
        st.dataframe(df,use_container_width=True,hide_index=True)
    st.download_button("⬇ Download results as CSV",df.to_csv(index=False).encode("utf-8-sig"),file_name="language_token_analysis.csv",mime="text/csv")