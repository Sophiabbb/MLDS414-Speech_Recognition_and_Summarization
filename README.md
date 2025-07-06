# Speech Recognition and Summarization

## Project Overview

This project combines Automatic Speech Recognition (ASR) and Named Entity Recognition (NER) with multi-level summarization to process and analyze spoken content. The pipeline transcribes audio using Whisper (OpenAI), extracts name entities with spaCy, and generates summaries at varying lengths (sentence, paragraph, multi-paragraph) using fine-tuned T5-Small and LLM-based agents. A Streamlit web app provides an interactive interface for users to upload audio, extract name entities, and obtain summaries.

---

## Speech Recognition and Named Entity Recognition Approach

### 1. Data Collection
We used [LibriSpeech](https://www.openslr.org/12/) dataset, a corpus of approximately 1000 hours of 16kHz English speech derived from LibriVox audiobooks. The dataset is carefully segmented and aligned, and it includes both the audio recordings and the original text that the audio narrates.

### 2. Automatic Speech Recognition (ASR)
- Used **Whisper** to transcribe audio clips from LibriSpeech. Whisper is a pre-trained and open-source ASR model developed by OpenAI.
- Evaluated transcription quality by comparing Whisper-generated transcripts to the original text using ROUGE scores. Whisper gives high-quality transcriptions that match well with the original text.

### 3. Exploratory Data Analysis (EDA)
- Analyzed and compared the distributions of audio duration, word count, and token length between the original text and the Whisper-generated transcripts.

### 4. Named Entity Recognition (NER)
- Applied spaCy’s `en_core_web_trf` model to extract named entities including person, organization, location, date, time, etc.

---

## Summarization Approach

We experimented with different summarization models and methods to generate summaries from the Whisper-generated transcripts at three levels of length:
  - Long summary (multiple paragraphs)
  - Short summary (one paragraph)
  - Tiny summary (one sentence)

### Model 1: Pretrained T5-Small
- Used the pretrained **T5-Small** model to summarize Whisper-generated transcripts without any task-specific fine-tuning.
- The model struggled to generate concise or meaningful summaries, often producing shortened excerpts of the original text with minimal abstraction.

### Model 2: Fine-Tuned T5-Small
- Used **LLaMA 3 8B Instruct** with **prompt engineering** to create high-quality reference summaries.
- Fine-tuned T5-Small using **LoRA (Low-Rank Adaptation)**.
- The fine-tuned T5-Small model produced significantly more accurate, relevant, and informative summaries compared to the pretrained version.

### Model 3: LLM-Based Agents
- Developed an agentic summarization system using **LLaMA 3 8B Instruct**, composed of three collaborative agents:
  - **Summarizer**: Generates an initial summary from the input text using prompt instructions and **in-context learning (ICL)** examples.
  - **Evaluator**: Assesses the summary’s quality, provides feedback, and determines whether it passes or fails.
  - **Regenerator**: Revises the summary based on the evaluator's feedback if the summary doesn't pass the evaluation.

### Summary Evaluation and Models Comparision
- Evaluation methods:
  - **BERTScore**: Captures semantic similarity between the original and summarized text using contextual embeddings.
  - **LLM-as-a-Judge**:
    - Fluency score: Assesses grammar and readability.
    - Coverage score: Evaluates inclusion of key information.
    - Coherence score: Measures logical flow and structural clarity.
    - Faithfulness score: Checks factual consistency with the original text.
- Results: LLM-based agents significantly outperformed both the fine-tuned T5 and the original T5 models across all summary lengths (Tiny, Short, long).

---

## Streamlit Web App
The project includes an interactive Streamlit web application that users can:
- Upload audio files and generate transcriptions
- Extract named entities of the transcription
- Generate summaries at multiple abstraction levels (Tiny, Short, long)

---

## Run Instructions

### 1. Clone the Repository
```bash
git git@github.com:Sophiabbb/MLDS414-Speech_Recognition_and_Summarization.git
cd MLDS414-Speech_Recognition_and_Summarization/web_app
```

### 2. Environment Setup
You must request access to the Meta-Llama-3-8B-Instruct model on Hugging Face and receive approval before using it.  
After approval, create a `.env` file in the `web_app` directory and add your Hugging Face token:

```env
HUGGINGFACE_TOKEN=your_token_here
```
Replace `your_token_here` with your actual Hugging Face token.

### 3. Build the Docker Image
```bash
docker build -f Dockerfile -t audio_summary .
```

### 4. Run the Application
```bash
docker run -p 8501:8501 --env-file .env audio_summary
```

### 5. Access the Application
Open your browser and go to:  
[http://localhost:8501](http://localhost:8501)

---

## Files Description
```
├── data/                      # Model outputs and evaluation results
├── t5_fine_tuning/            # Scripts for LoRA fine-tuning
├── web_app/                   # Streamlit web application files
│   ├── LoRA_Weights           # Fine-tuned LoRA weights for T5-small
│   ├── Dockerfile             # Docker configuration
│   ├── processing.py          # Preprocessing and summarization logic for the app
│   ├── requirements.txt       # Python dependencies
│   └── webapp.py              # Main entry point for the Streamlit interface
├── 1_Speech_to_Text.ipynb     # ASR, NER, and EDA notebook
├── 2_Text_Summarization.ipynb # Summarization modeling notebook
├── 3_Summary_Evaluation.ipynb # Summary evaluation notebook
├── README.md                  # Main documentation for the project
└── Speech Recognition and Summarization.pdf # Final presentation slides
```

---

## Contributors
- Fuqian Zou  
- Glenys Charity Lion  
- Iris Lee  
- Kavya Bhat  
- Liana Bergman-Turnbull
