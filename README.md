# News Summarization and LLM Retraining Project

This project implements an end-to-end pipeline for collecting news articles, generating summaries using Google's Gemini API, and retraining a language model on the summarization task.

## 🌟 Features

- RSS feed parsing from multiple news sources
- Web scraping with intelligent content extraction
- Automated news summarization using Google Gemini API
- Dataset creation for fine-tuning
- LLM retraining pipeline using Unsloth optimization

## 📋 Project Structure

```
.
├── news_summarizer.py     # Main summarization pipeline
├── parser.py             # RSS feed parsing utilities
├── scrapper.py          # Web scraping functionality
├── sources.json         # RSS feed source configurations
└── retrain.ipynb       # LLM fine-tuning notebook
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Variables

Set up your API key for Google Gemini:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## 💡 Usage

### 1. Collecting and Summarizing News

The pipeline starts with `news_summarizer.py`:

```python
from news_summarizer import NewsSummarizer

summarizer = NewsSummarizer(api_key="your-api-key")
summaries = await summarizer.process_feed("feed_url", hours=24)
```

### 2. Training Data Generation

The system will:
- Parse RSS feeds from configured sources
- Extract full article content
- Generate summaries using Gemini API
- Save the dataset in a structured format

### 3. Model Retraining

Run the `retrain.ipynb` notebook to:
- Load the generated dataset
- Configure the Unsloth-optimized model
- Fine-tune on the summarization task
- Save the trained model

## 🔧 Configuration

### RSS Sources

Edit `sources.json` to configure news sources:

```json
{
    "source_name": "rss_feed_url",
    "El_Pais": "https://feeds.elpais.com/mrss-s/...",
    ...
}
```

### Training Configuration

Key parameters in `retrain.ipynb`:

```python
MODEL_NAME = "unsloth/Llama-3.2-3B-Instruct"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
```

## 🛠️ Technical Details

### Scraping Engine

- Intelligent content extraction using BeautifulSoup4
- Image quality assessment and filtering
- Robust error handling and retry mechanisms
- Rate limiting with random jitter

### Summarization

- Bullet-point format summaries
- Multi-language support (maintains source article language)
- Configurable safety settings
- Quota management and exponential backoff

### Training Pipeline

- LoRA fine-tuning configuration
- Unsloth optimization for 2x faster training
- Mixed precision training
- Gradient accumulation
- TensorBoard logging

## 📊 Performance Optimization

- Batch processing for RSS feeds
- Asynchronous operations
- Intelligent rate limiting
- Memory-efficient data handling
- GPU optimization with Unsloth

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🙏 Acknowledgments

- Google Gemini API for summarization
- Unsloth team for training optimizations
- Scrapper and parser libraries from my [ReMarkNews](https://github.com/frrobledo/ReMarkNews) project