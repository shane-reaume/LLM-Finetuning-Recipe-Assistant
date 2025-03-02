---
language: en
license: mit
datasets:
- imdb
metrics:
- accuracy
- f1
tags:
- sentiment-analysis
- text-classification
- distilbert
- movie-reviews
pipeline_tag: text-classification
widget:
- text: "This movie was absolutely amazing, I loved every minute of it!"
- text: "The acting was terrible and the plot made no sense at all."
- text: "While it had some good moments, overall I was disappointed."
---

# DistilBERT for Sentiment Analysis - IMDB Movie Reviews

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the [IMDB dataset](https://huggingface.co/datasets/imdb), trained for sentiment analysis of movie reviews.

## Model Details

- **Developed by:** Shane Reaume ([@shane-reaume](https://huggingface.co/shane-reaume))
- **Model type:** Fine-tuned DistilBERT
- **Language:** English
- **License:** MIT
- **Finetuned from:** [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

## Performance

Evaluation on a balanced test set of 100 examples (50 positive, 50 negative):

| Metric | Value |
|--------|-------|
| Accuracy | 84.00% |
| F1 Score | 0.8462 |
| Precision | 81.48% |
| Recall | 88.00% |
| Inference Time | ~3.30 ms/example |

## Uses

### Direct Use

This model can be used directly for sentiment analysis of text, particularly movie reviews:
