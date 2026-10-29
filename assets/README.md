# 🔤 Indian Language Token Analysis

<p align="center">
  <b>Multilingual Translation & Tokenization Analysis Using NLP</b>
</p>

<p align="center">
  Detect the input language, translate text into English, Hindi, Bengali, and Tamil, and compare token counts and processing time using local and cloud-based models.
</p>

<p align="center">
  <a href="https://indian-language-token-analysis-test.streamlit.app/">🚀 Live Demo</a> •
  <a href="https://github.com/ShivamDevHere/indian-language-token-analysis-test">💻 Source Code</a> •
  <a href="https://github.com/ShivamDevHere">👨‍💻 My GitHub</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/NLP-Multilingual-blue?style=for-the-badge" alt="NLP"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/>
</p>

---
## Problem Statement
AI splits text into small pieces called tokens. Some Indian languages need more pieces, making answers slower and more expensive. Splitting can also hide harmful phrases from safety checks. Developers need to compare speed, cost and safety before choosing a model.

## 📌 About the Project

**Indian Language Token Analysis** is an NLP application that detects the language of input text, translates it into four supported languages, and analyzes the resulting text using the MuRIL tokenizer.

Supported languages:

- **English**
- **Hindi**
- **Bengali**
- **Tamil**

The application supports local translation with NLLB-200 and cloud-based translation through Gemini, OpenAI, and Groq APIs.

## ✨ Key Features

- 🌐 **Language Detection** — Identifies the input language automatically.
- 🔄 **Multilingual Translation** — Translates text into four supported languages.
- 🧠 **Multiple AI Providers** — Supports NLLB-200, Gemini, OpenAI, and Groq.
- 🔢 **Token Analysis** — Counts tokens using the MuRIL tokenizer.
- ⏱️ **Performance Measurement** — Records translation and tokenization time.
- ⚡ **Automatic Fallback** — Offers switching to local translation when a cloud request fails.
- 📊 **Interactive Dashboard** — Displays translations, metrics, and token comparisons.
- 📥 **CSV Export** — Downloads analysis results for further examination.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Web application interface |
| Lingua | Language detection |
| Hugging Face Transformers | Model loading and tokenization |
| NLLB-200 | Local machine translation |
| MuRIL | Tokenization of translated text |
| PyTorch | Deep learning inference |
| Pandas | Data processing and analysis |
| Gemini API | Cloud-based translation |
| OpenAI API | Cloud-based translation |
| Groq API | Cloud-based translation |

## 🧠 Models Used

### 1. NLLB-200

- **Model:** `facebook/nllb-200-distilled-600M`
- Performs translation locally.
- Does not require cloud API credentials.
- Requires downloading the model and sufficient system memory.

### 2. MuRIL

- **Model:** `google/muril-base-cased`
- Tokenizes translated text.
- Enables token-count comparisons across supported languages.

### 3. Cloud Translation APIs

- **Gemini API** — Google's generative AI models.
- **OpenAI API** — OpenAI's language models.
- **Groq API** — Language models served through Groq's API.

Cloud providers require valid API keys, and availability depends on model access, quotas, and account limits.

## ⚙️ How It Works

```text
User Input
    ↓
Language Detection (Lingua)
    ↓
Select Translation Provider
    ↓
Translate into English, Hindi, Bengali, and Tamil
    ↓
Tokenization using MuRIL
    ↓
Measure Translation and Tokenization Time
    ↓
Display Results and Comparison Chart
    ↓
Export Results as CSV
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or newer
- Git
- Internet connection for installing dependencies and downloading models
- API keys for the cloud providers you want to use

### 1. Clone the Repository

```bash
git clone https://github.com/ShivamDevHere/indian-language-token-analysis-test.git
cd indian-language-token-analysis-test
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
```

Get API keys from:

- [Google AI Studio](https://aistudio.google.com/)
- [OpenAI Platform](https://platform.openai.com/)
- [Groq Console](https://console.groq.com/)

For Streamlit Community Cloud, configure these keys in your app's **Settings → Secrets** instead of uploading the `.env` file.

**Security:** Never commit API keys or your `.env` file to GitHub.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

> **NOTE:** If a cloud translation request fails, the application offers a fallback to Local NLLB-200.

## 📊 Analysis Metrics

| Metric | Description |
|---|---|
| Characters | Number of characters in the translated text |
| Tokens | Number of tokens produced by MuRIL |
| Translation Time | Time taken to generate a translation |
| Tokenization Time | Time taken to tokenize translated text |
| Model Used | Translation provider used for the result |

> **Note:** Token counts depend on the tokenizer. MuRIL token counts are not necessarily comparable to tokens billed by cloud APIs. Translation time includes API and network latency for cloud providers.

## 📂 Project Structure

```text
indian-language-token-analysis/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## 🌐 Project Links

- 🚀 **Live Demo:** [Open Application](https://indian-language-token-analysis-test.streamlit.app/)
- 💻 **Source Code:** [GitHub Repository](https://github.com/ShivamDevHere/indian-language-token-analysis-test)
- 👨‍💻 **GitHub Profile:** [@ShivamDevHere](https://github.com/ShivamDevHere)

## 🎯 Project Objective

To explore multilingual NLP by analyzing translation outputs, token counts, and processing performance across four languages using local and cloud-based translation models.

## 👨‍💻 Author

**Shivam Srivastava**

- GitHub: [@ShivamDevHere](https://github.com/ShivamDevHere)
- LinkedIn: [Connect with me](https://linkedin.com/in/shivamdevhere)
- Portfolio: [Visit My Portfolio](https://portfolio-ruby-one-66.vercel.app/)

---

<p align="center">
  ⭐ If you find this project useful, consider giving it a star!
</p>

<p align="center">
  <i>Built to explore multilingual NLP, machine translation, and tokenization analysis.</i>
</p>
