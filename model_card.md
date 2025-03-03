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

- **Developed by:** YOUR_NAME ([@YOUR_HUGGINGFACE_USERNAME](https://huggingface.co/YOUR_HUGGINGFACE_USERNAME))
- **Model type:** Fine-tuned DistilBERT
- **Language:** English
- **License:** MIT
- **Finetuned from:** [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased)

## Performance

Evaluation on a balanced test set of 100 examples (50 positive, 50 negative):

| Metric | Value |
|--------|-------|
| Accuracy | XX.XX% |
| F1 Score | X.XXXX |
| Precision | XX.XX% |
| Recall | XX.XX% |
| Inference Time | ~X.XX ms/example |

## Uses

### Direct Use

This model can be used directly for sentiment analysis of text, particularly movie reviews:

```python
from transformers import pipeline

# Load the model
sentiment = pipeline("sentiment-analysis", model="YOUR_USERNAME/YOUR_MODEL_NAME")

# Make predictions
result = sentiment("I really enjoyed this movie!")
print(result)
```

### Example Results

```
[{'label': 'POSITIVE', 'score': 0.9998}]  # For positive text
[{'label': 'NEGATIVE', 'score': 0.9987}]  # For negative text
```

## Training Details

- **Dataset:** IMDB Movie Reviews dataset (25,000 training examples)
- **Base Model:** distilbert-base-uncased
- **Training Parameters:**
  - Learning rate: 2e-5
  - Batch size: 16
  - Epochs: 3
  - Optimizer: AdamW

## Limitations and Biases

This model was trained on movie reviews and may not perform as well on other types of text. It may also reflect biases present in the training data.

## Citation

```
@inproceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```
