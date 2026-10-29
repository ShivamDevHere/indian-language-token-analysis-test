# 🔤 Indian Language Token Analysis

A Streamlit-based application that detects Indian languages, translates text, and compares tokenization across **English, Hindi, Bengali, and Tamil** using NLP models.

## 🚀 Features

- **Language Detection:** Automatically identifies the input language.
- **Multi-Language Translation:** Translates text into four supported languages.
- **Multiple Translation Models:** Choose Local NLLB-200, Gemini API, or OpenAI API.
- **Token Analysis:** Counts tokens using the MuRIL tokenizer.
- **Performance Measurement:** Measures translation and tokenization time.
- **Live Timer:** Displays elapsed time during cloud translation.
- **Automatic Fallback:** Switches to Local NLLB-200 if a cloud request fails or times out.
- **Interactive Dashboard:** Displays translations, token counts, and performance metrics.
- **CSV Export:** Download analysis results for further examination.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| Lingua | Language detection |
| Hugging Face Transformers | NLP models and tokenization |
| NLLB-200 | Local machine translation |
| MuRIL | Tokenization for Indian languages |
| PyTorch | Deep learning inference |
| Pandas | Results processing and tabular analysis |
| Gemini API | Cloud-based translation |
| OpenAI API | Cloud-based translation |

## 🧠 Models Used

### 1. NLLB-200
- Model: `facebook/nllb-200-distilled-600M`
- Performs translation locally.
- Does not require paid API access.
- Requires downloading the model and sufficient system memory.

### 2. MuRIL
- Model: `google/muril-base-cased`
- Tokenizes the translated text.
- Helps compare token counts across supported languages.

### 3. Gemini API
- Uses Google's Gemini models for cloud translation.
- Requires a valid API key and available quota.

### 4. OpenAI API
- Uses OpenAI's API for cloud translation.
- Requires a valid API key and available API billing or credits.

### 5. Groq API
- Uses Groq's API for cloud translation.
- Requires a valid API key and available API freely.

## ⚙️ Installation

### Prerequisites

- Python 3.10 or newer
- Git
- Internet connection for installation and model downloads
- API keys only if using Gemini or OpenAI

### 1. Clone the repository

```bash
git clone https://github.com/ShivamDevHere/indian-language-token-analysis-test.git
cd indian-language-token-analysis-test
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys (optional)

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

Get API access from:

- [Google AI Studio](https://aistudio.google.com/)
- [OpenAI Platform](https://platform.openai.com/)
- [Groq Platform](https://console.groq.com/)
- 

**Security:** Never commit your `.env` file or expose API keys publicly.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📖 How to Use

1. Enter text in English, Hindi, Bengali, or Tamil.
2. Select a translation model.
3. Click **Analyze**.
4. View the detected source language and translations.
5. Compare token counts, translation time, and tokenization time.
6. View the token-count comparison chart.
7. Download the results as a CSV file.

If a cloud translation request fails, the application provides a fallback option to use Local NLLB-200.

## 📊 Analysis Metrics

| Metric | Description |
|---|---|
| Characters | Number of characters in the translated text |
| Tokens | Number of tokens produced by the MuRIL tokenizer |
| Translation Time | Time taken to produce a translation |
| Tokenization Time | Time taken to tokenize the translated text |
| Model Used | Translation model used for the result |

**Note:** Token counts depend on the tokenizer. MuRIL token counts are not necessarily comparable to the token counts billed by Gemini or OpenAI. Cloud translation time includes network and API latency.

## 🏗️ Project Workflow

```text
User Input
    ↓
Language Detection (Lingua)
    ↓
Select Translation Model
    ↓
Translate into English, Hindi, Bengali, and Tamil
    ↓
Tokenization using MuRIL
    ↓
Measure Processing Time
    ↓
Display Results and Comparison Chart
    ↓
Export Results as CSV
```

## 📁 Project Structure

```text
indian-language-token-analysis/
│
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore          # Files excluded from Git
└── README.md           # Project documentation
```

## 🔐 Environment and API Notes

- Local NLLB-200 works without cloud API credentials.
- Gemini and OpenAI depend on API availability, quota, and account configuration.
- Cloud requests have timeout handling; local inference may take longer on the first run.
- API pricing and free-tier limits may change.
- The application measures translation and tokenization performance; it does not currently calculate actual API charges.

## 🎯 Project Objective

To explore multilingual NLP by comparing translation performance and tokenization across four languages, while providing a simple interface for experimenting with local and cloud-based translation models.

## 👨‍💻 Author

**Shivam Srivastava**

- GitHub: [ShivamDevHere](https://github.com/ShivamDevHere)
- LinkedIn: [shivamdevhere](https://linkedin.com/in/shivamdevhere)
- Portfolio: [Visit Portfolio](https://portfolio-ruby-one-66.vercel.app/)

---

*Built to explore multilingual NLP, machine translation, and tokenization analysis.*
