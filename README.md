# Phishing Email Detection Study

A notebook-based study comparing engineered email features and machine-learning classifiers with a custom multilingual DistilBERT classifier. It includes corpus exploration, feature extraction, classical-model experiments, and transformer training/export code.

## Repository scope

The main artifact is `Phishing_Email_Detection_Deliverable_Complete.ipynb`. The CSV corpora, Google Drive directory, and exported model files referenced by the notebook are **not included**. This repository is a research workflow rather than a ready-to-run email gateway or hosted detector.

## Open the notebook

[Open in Google Colab](https://colab.research.google.com/github/AbdelilahNossair/Phishing-Email-Detection/blob/main/Phishing_Email_Detection_Deliverable_Complete.ipynb)

For local use:

```bash
git clone https://github.com/AbdelilahNossair/Phishing-Email-Detection.git
cd Phishing-Email-Detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install jupyterlab pandas numpy scikit-learn torch matplotlib seaborn wordcloud
python -m pip install transformers==4.30.2 accelerate==0.21.0
jupyter lab Phishing_Email_Detection_Deliverable_Complete.ipynb
```

The two version pins reproduce the notebook's installation cells; they are historical dependencies, not a complete environment lock. Additional model-specific imports should be installed as needed for the sections you execute. A GPU and sufficient memory are recommended for transformer training.

## Required data

The notebook reads files from `/content/drive/MyDrive/Phishing_Email_Detection/`:

| File | Corpus |
| --- | --- |
| `CEAS_08.csv` | CEAS |
| `SpamAssasin.csv` | SpamAssassin; preserve the filename spelling used in code |
| `cleaned_TREC_06.csv` | TREC 2006 |
| `cleaned_TREC_05.csv` | TREC 2005 |
| `cleaned_TREC_07.csv` | TREC 2007 |
| `Enron.csv` | Enron |
| `Ling.csv` | Ling |
| `Nazario.csv` | Nazario |
| `Nigerian_Fraud.csv` | Nigerian fraud collection |

Obtain authorized copies, normalize their schemas, and edit the paths for your environment. In Colab, mount the appropriate Drive explicitly if it is not already mounted. Exploration uses `subject`, `body`, and `label`, with notebook labels 0 for legitimate and 1 for phishing. Confirm label semantics for every source: spam and phishing are not interchangeable by default.

## Workflow

1. Explore corpus sizes, missing values, class balance, message lengths, word clouds, URLs, special characters, and urgency indicators.
2. Combine datasets and construct numeric/boolean signals for suspicious or generic language, credential requests, formatting, attachments, and related patterns.
3. Split features/labels and compare classical classifiers and tuning configurations.
4. Prepare combined text and additional features for `distilbert-base-multilingual-cased`.
5. Concatenate the transformer representation with engineered features in a custom PyTorch classifier.
6. Train with custom Trainer/loss code and compute classification metrics.
7. Save the model state dictionary, tokenizer, and configuration to Google Drive.

## Export and inference implications

The export targets `models/phishing_detector_v2` under the notebook's Drive directory. Files include `pytorch_model.bin`, tokenizer files, and `model_config.bin`.

The classifier is a custom `nn.Module`. Loading its state dictionary requires the matching class definition, feature order/dimensions, and preprocessing; it is not automatically equivalent to a stock Hugging Face sequence-classification checkpoint.

## Important reproducibility limits

The notebook contains repeated/revised definitions and multiple training sections. Run a reviewed sequence from a clean kernel rather than mixing cells from different iterations. An early branch constructs `balanced_df` but later extracts features from `combined_df`; confirm which dataframe actually reaches each training run.

Before interpreting scores, check duplicates across corpora and train/test partitions, preserve corpus provenance, and evaluate on an unseen source. A random mixed-corpus split can overestimate generalization. No freshly reproduced performance claim is made here.

Use sanitized samples when publishing outputs. No standalone license or dataset provenance manifest is included; corpus and model usage terms require separate review.
