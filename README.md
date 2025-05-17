# Comparative Evaluation of Lightweight Pretrained Language Models for Fake News Detection

## Project Overview

This project investigates the effectiveness of lightweight pretrained language models for fake news detection, focusing on their performance, efficiency, and practical applicability. The research compares several lightweight transformer-based models (TinyBERT, DistilBERT, MobileBERT, and RoBERTa) against traditional machine learning baselines to determine their suitability for deployment in resource-constrained environments.

The project utilizes the ISOT Fake News Dataset for training and evaluation, with comprehensive data analysis, model fine-tuning, and performance evaluation across multiple metrics including accuracy, precision, recall, F1-score, and computational efficiency.

## Project Structure

The project is organized into the following main directories:

### Data Analysis
Contains exploratory data analysis of the ISOT dataset:
- `EDA_ISOT.ipynb`: Exploratory data analysis notebook
- `ISOT_analysis.ipynb`: In-depth analysis of dataset characteristics

### Data
Contains the ISOT dataset and related files:
- `ISOT/`: ISOT Fake News Dataset files
- `collect_reuters.py`: Script for collecting Reuters news articles
- `create_fake.py`: Script for processing fake news data

### Baseline
Contains traditional machine learning baseline models:
- `Traditional_Machine_Learning_Baselines_for_Fake_News_Detection.ipynb`: Implementation of baseline models
- `Traditional_ML_Baseline_Results_Analysis.ipynb`: Analysis of baseline model results

### Finetuning
Contains notebooks for fine-tuning the lightweight language models:
- `tinyBERT/`: TinyBERT model fine-tuning
- `distilBERT/`: DistilBERT model fine-tuning
- `mobileBERT/`: MobileBERT model fine-tuning
- `roBERTa/`: RoBERTa model fine-tuning

### Evaluation
Contains evaluation scripts and results for each model:
- `TinyBERT_Eval.ipynb`: Evaluation of TinyBERT model
- `DistillBERT_Eval.ipynb`: Evaluation of DistilBERT model
- `MobileBERT_Eval.ipynb`: Evaluation of MobileBERT model
- `RoBERTa_eval.ipynb`: Evaluation of RoBERTa model
- `datasets/`: Additional evaluation datasets
- `process_dataset.py`: Script for processing evaluation datasets

### Comparative Analysis
Contains comparative analysis of all models:
- `Comparative_Analysis_of_Models_for_Fake_News_Detection.ipynb`: Comprehensive comparison of all models

### Documentation
Contains project documentation:
- `Research_Proposal.pdf`: Original research proposal
- `Final_Report.pdf`: Final project report

## Installation and Setup

To run the notebooks and scripts in this project, you need to set up the required environment:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Comparative-Evaluation-of-Lightweight-Pretrained-Language-Models-for-Fake-News-Detection.git
cd Comparative-Evaluation-of-Lightweight-Pretrained-Language-Models-for-Fake-News-Detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Download the ISOT Fake News Dataset (if not already included):
   - The dataset should be placed in the `data/ISOT/` directory
   - Alternatively, you can use the provided scripts to collect and process data

2. Run the data analysis notebooks to understand the dataset characteristics:
```bash
jupyter notebook data_analysis/EDA_ISOT.ipynb
```

### Running Baseline Models

1. Execute the traditional machine learning baseline notebook:
```bash
jupyter notebook baseline/Traditional_Machine_Learning_Baselines_for_Fake_News_Detection.ipynb
```

2. Analyze the baseline results:
```bash
jupyter notebook baseline/Traditional_ML_Baseline_Results_Analysis.ipynb
```

### Fine-tuning Language Models

To fine-tune each of the lightweight language models:

1. TinyBERT:
```bash
jupyter notebook finetuning/tinyBERT/TinyBERT_ISOT.ipynb
```

2. DistilBERT (currently being fine-tuned with cleaned data):
```bash
jupyter notebook finetuning/distilBERT/DistilBERT_ISOT.ipynb
```

3. MobileBERT (currently being fine-tuned with cleaned data):
```bash
jupyter notebook finetuning/mobileBERT/MobileBERT_ISOT.ipynb
```

4. RoBERTa (currently being fine-tuned with cleaned data):
```bash
jupyter notebook finetuning/roBERTa/RoBERTa_ISOT.ipynb
```

### Model Evaluation

To evaluate the performance of each model:

1. TinyBERT:
```bash
jupyter notebook evaluation/TinyBERT_Eval.ipynb
```

2. DistilBERT:
```bash
jupyter notebook evaluation/DistillBERT_Eval.ipynb
```

3. MobileBERT:
```bash
jupyter notebook evaluation/MobileBERT_Eval.ipynb
```

4. RoBERTa:
```bash
jupyter notebook evaluation/RoBERTa_eval.ipynb
```

### Comparative Analysis

To compare the performance of all models:
```bash
jupyter notebook comparative_analysis/Comparative_Analysis_of_Models_for_Fake_News_Detection.ipynb
```

## Key Findings

The project compares lightweight transformer-based models against traditional machine learning approaches for fake news detection. Key findings include:

1. Performance comparison between lightweight transformer models and traditional machine learning baselines
2. Analysis of computational efficiency and resource requirements
3. Trade-offs between model size, inference speed, and detection accuracy
4. Practical considerations for deploying these models in resource-constrained environments

Note: The fine-tuning of DistilBERT, MobileBERT, and RoBERTa is currently in progress with cleaned data, so their final results are not yet included in the comparative analysis.

## Future Work

Potential areas for future research include:

1. Exploring additional lightweight language models
2. Testing on diverse fake news datasets beyond ISOT
3. Investigating domain adaptation techniques for improved performance
4. Developing ensemble methods combining multiple lightweight models
5. Exploring knowledge distillation to create even more efficient models

## Contact

For questions or further information about this project, please contact the project author.

