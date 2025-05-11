# Part 3: Fine-tuning RoBERTa for Fake News Detection

In this notebook, I'll build on our previous work with DistilBERT and TinyBERT to fine-tune a RoBERTa model for fake news detection. RoBERTa (Robustly Optimized BERT Pretraining Approach) is an enhanced version of BERT that improves upon it by training longer with bigger batches and more data, removing the next sentence prediction objective, and using dynamic masking patterns. These improvements often lead to better performance on downstream tasks.

Given that both DistilBERT and TinyBERT achieved excellent results, I want to explore if RoBERTa can provide even better performance or more robust generalization to unseen examples. Additionally, RoBERTa is designed to be more robust to overfitting, which could be particularly valuable for fake news detection where the model shouldn't rely on specific patterns in the training data.

## 1. Setup and Library Installation

First, I'll install the required packages:


```python
# Install required packages
!pip install transformers datasets torch evaluate scikit-learn
```

    Requirement already satisfied: transformers in /usr/local/lib/python3.11/dist-packages (4.51.1)
    Requirement already satisfied: datasets in /usr/local/lib/python3.11/dist-packages (3.5.0)
    Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (2.5.1+cu124)
    Collecting evaluate
      Downloading evaluate-0.4.3-py3-none-any.whl.metadata (9.2 kB)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (1.2.2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from transformers) (3.18.0)
    Requirement already satisfied: huggingface-hub<1.0,>=0.30.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.30.2)
    Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (1.26.4)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (24.2)
    Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from transformers) (6.0.2)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (2024.11.6)
    Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from transformers) (2.32.3)
    Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.21.0)
    Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.5.2)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.11/dist-packages (from transformers) (4.67.1)
    Requirement already satisfied: pyarrow>=15.0.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (19.0.1)
    Requirement already satisfied: dill<0.3.9,>=0.3.0 in /usr/local/lib/python3.11/dist-packages (from datasets) (0.3.8)
    Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (from datasets) (2.2.3)
    Requirement already satisfied: xxhash in /usr/local/lib/python3.11/dist-packages (from datasets) (3.5.0)
    Requirement already satisfied: multiprocess<0.70.17 in /usr/local/lib/python3.11/dist-packages (from datasets) (0.70.16)
    Collecting fsspec<=2024.12.0,>=2023.1.0 (from fsspec[http]<=2024.12.0,>=2023.1.0->datasets)
      Downloading fsspec-2024.12.0-py3-none-any.whl.metadata (11 kB)
    Requirement already satisfied: aiohttp in /usr/local/lib/python3.11/dist-packages (from datasets) (3.11.16)
    Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.11/dist-packages (from torch) (4.13.1)
    Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from torch) (3.4.2)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.6)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch) (12.4.127)
    Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch) (12.4.127)
    Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch) (12.4.127)
    Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch)
      Downloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cublas-cu12==12.4.5.8 (from torch)
      Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cufft-cu12==11.2.1.3 (from torch)
      Downloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-curand-cu12==10.3.5.147 (from torch)
      Downloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch)
      Downloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch)
      Downloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl.metadata (1.6 kB)
    Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch) (2.21.5)
    Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.11/dist-packages (from torch) (12.4.127)
    Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch)
      Downloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl.metadata (1.5 kB)
    Requirement already satisfied: triton==3.1.0 in /usr/local/lib/python3.11/dist-packages (from torch) (3.1.0)
    Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch) (1.13.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch) (1.3.0)
    Requirement already satisfied: scipy>=1.3.2 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.15.2)
    Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (1.4.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn) (3.6.0)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (2.6.1)
    Requirement already satisfied: aiosignal>=1.1.2 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.3.2)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.5.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (6.2.0)
    Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (0.3.1)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.11/dist-packages (from aiohttp->datasets) (1.19.0)
    Requirement already satisfied: mkl_fft in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers) (1.3.8)
    Requirement already satisfied: mkl_random in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers) (1.2.4)
    Requirement already satisfied: mkl_umath in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers) (0.1.1)
    Requirement already satisfied: mkl in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers) (2025.1.0)
    Requirement already satisfied: tbb4py in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers) (2022.1.0)
    Requirement already satisfied: mkl-service in /usr/local/lib/python3.11/dist-packages (from numpy>=1.17->transformers) (2.4.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.4.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2025.1.31)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch) (3.0.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas->datasets) (2025.2)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas->datasets) (1.17.0)
    Requirement already satisfied: intel-openmp<2026,>=2024 in /usr/local/lib/python3.11/dist-packages (from mkl->numpy>=1.17->transformers) (2024.2.0)
    Requirement already satisfied: tbb==2022.* in /usr/local/lib/python3.11/dist-packages (from mkl->numpy>=1.17->transformers) (2022.1.0)
    Requirement already satisfied: tcmlib==1.* in /usr/local/lib/python3.11/dist-packages (from tbb==2022.*->mkl->numpy>=1.17->transformers) (1.2.0)
    Requirement already satisfied: intel-cmplr-lib-rt in /usr/local/lib/python3.11/dist-packages (from mkl_umath->numpy>=1.17->transformers) (2024.2.0)
    Requirement already satisfied: intel-cmplr-lib-ur==2024.2.0 in /usr/local/lib/python3.11/dist-packages (from intel-openmp<2026,>=2024->mkl->numpy>=1.17->transformers) (2024.2.0)
    Downloading nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m363.4/363.4 MB[0m [31m4.7 MB/s[0m eta [36m0:00:00[0m0:00:01[0m00:01[0m
    [?25hDownloading nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m664.8/664.8 MB[0m [31m2.5 MB/s[0m eta [36m0:00:00[0m0:00:01[0m00:01[0m
    [?25hDownloading nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m211.5/211.5 MB[0m [31m8.0 MB/s[0m eta [36m0:00:00[0m0:00:01[0m00:01[0m
    [?25hDownloading nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m56.3/56.3 MB[0m [31m26.5 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m127.9/127.9 MB[0m [31m9.6 MB/s[0m eta [36m0:00:00[0m0:00:01[0m00:01[0m
    [?25hDownloading nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m207.5/207.5 MB[0m [31m8.1 MB/s[0m eta [36m0:00:00[0m0:00:01[0m00:01[0m
    [?25hDownloading nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m21.1/21.1 MB[0m [31m74.2 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hDownloading evaluate-0.4.3-py3-none-any.whl (84 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m84.0/84.0 kB[0m [31m6.2 MB/s[0m eta [36m0:00:00[0m
    [?25hDownloading fsspec-2024.12.0-py3-none-any.whl (183 kB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m183.9/183.9 kB[0m [31m14.8 MB/s[0m eta [36m0:00:00[0m
    [?25hInstalling collected packages: nvidia-nvjitlink-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cublas-cu12, fsspec, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, evaluate
      Attempting uninstall: nvidia-nvjitlink-cu12
        Found existing installation: nvidia-nvjitlink-cu12 12.8.93
        Uninstalling nvidia-nvjitlink-cu12-12.8.93:
          Successfully uninstalled nvidia-nvjitlink-cu12-12.8.93
      Attempting uninstall: nvidia-curand-cu12
        Found existing installation: nvidia-curand-cu12 10.3.9.90
        Uninstalling nvidia-curand-cu12-10.3.9.90:
          Successfully uninstalled nvidia-curand-cu12-10.3.9.90
      Attempting uninstall: nvidia-cufft-cu12
        Found existing installation: nvidia-cufft-cu12 11.3.3.83
        Uninstalling nvidia-cufft-cu12-11.3.3.83:
          Successfully uninstalled nvidia-cufft-cu12-11.3.3.83
      Attempting uninstall: nvidia-cublas-cu12
        Found existing installation: nvidia-cublas-cu12 12.8.4.1
        Uninstalling nvidia-cublas-cu12-12.8.4.1:
          Successfully uninstalled nvidia-cublas-cu12-12.8.4.1
      Attempting uninstall: fsspec
        Found existing installation: fsspec 2025.3.2
        Uninstalling fsspec-2025.3.2:
          Successfully uninstalled fsspec-2025.3.2
      Attempting uninstall: nvidia-cusparse-cu12
        Found existing installation: nvidia-cusparse-cu12 12.5.8.93
        Uninstalling nvidia-cusparse-cu12-12.5.8.93:
          Successfully uninstalled nvidia-cusparse-cu12-12.5.8.93
      Attempting uninstall: nvidia-cudnn-cu12
        Found existing installation: nvidia-cudnn-cu12 9.3.0.75
        Uninstalling nvidia-cudnn-cu12-9.3.0.75:
          Successfully uninstalled nvidia-cudnn-cu12-9.3.0.75
      Attempting uninstall: nvidia-cusolver-cu12
        Found existing installation: nvidia-cusolver-cu12 11.7.3.90
        Uninstalling nvidia-cusolver-cu12-11.7.3.90:
          Successfully uninstalled nvidia-cusolver-cu12-11.7.3.90
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    gcsfs 2024.10.0 requires fsspec==2024.10.0, but you have fsspec 2024.12.0 which is incompatible.
    bigframes 1.36.0 requires rich<14,>=12.4.4, but you have rich 14.0.0 which is incompatible.
    pylibcugraph-cu12 24.12.0 requires pylibraft-cu12==24.12.*, but you have pylibraft-cu12 25.2.0 which is incompatible.
    pylibcugraph-cu12 24.12.0 requires rmm-cu12==24.12.*, but you have rmm-cu12 25.2.0 which is incompatible.[0m[31m
    [0mSuccessfully installed evaluate-0.4.3 fsspec-2024.12.0 nvidia-cublas-cu12-12.4.5.8 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nvjitlink-cu12-12.4.127


Now let's import the necessary libraries:


```python
# Import basic libraries
import numpy as np
import pandas as pd
import torch
import random
import time
import os
import warnings
```


```python
# Import transformers and datasets libraries
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from datasets import Dataset as HFDataset
```

    2025-05-11 09:40:38.631871: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1746956438.815969      31 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1746956438.867984      31 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered



```python
# Import evaluation libraries
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
```

Set up the environment and reproducibility:


```python
# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

    Using device: cuda


## 2. Load and Prepare the Dataset

Let's load the preprocessed datasets from our previous work:


```python
# Load the preprocessed datasets
try:
    train_df = pd.read_csv('/kaggle/input/isot-processed-and-splitted/train_fake_news.csv')
    val_df = pd.read_csv('/kaggle/input/isot-processed-and-splitted/val_fake_news.csv') 
    test_df = pd.read_csv('/kaggle/input/isot-processed-and-splitted/test_fake_news.csv')
    
    print(f"Training set: {train_df.shape}")
    print(f"Validation set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")
except FileNotFoundError:
    print("Preprocessed files not found. Please run the data preprocessing from Part 2 first.")
```

    Training set: (31428, 3)
    Validation set: (6735, 3)
    Test set: (6735, 3)


Let's examine the data format to ensure it's what we expect:


```python
# Display sample data
print("Sample of training data:")
train_df.head(3)
```

    Sample of training data:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>enhanced_cleaned_text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trump ‘Diversity Council’ Member Threatens to ...</td>
      <td>A member of President Trump s Diversity Counci...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DID BEYONCE AND JAY Z’s “Vacation” To Communis...</td>
      <td>Notorious radical Black Panther and NJ cop kil...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN Host Calls Out Trump’s Uncle Tom Spokeswo...</td>
      <td>Katrina Pierson is a black woman. She is also ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now I'll define a function to convert our pandas DataFrames to the Hugging Face Dataset format:


```python
# Function to convert pandas DataFrames to HuggingFace Datasets
def convert_to_hf_dataset(df):
    # For RoBERTa, we'll use both title and text
    df['text'] = df['title'] + " " + df['enhanced_cleaned_text']
    
    # Convert to HuggingFace Dataset format
    dataset = HFDataset.from_pandas(df[['text', 'label']])
    return dataset
```

Apply the conversion function to our datasets:


```python
# Convert our datasets
train_dataset = convert_to_hf_dataset(train_df)
val_dataset = convert_to_hf_dataset(val_df)
test_dataset = convert_to_hf_dataset(test_df)

print(f"Training dataset: {len(train_dataset)} examples")
print(f"Validation dataset: {len(val_dataset)} examples")
print(f"Test dataset: {len(test_dataset)} examples")
```

    Training dataset: 31428 examples
    Validation dataset: 6735 examples
    Test dataset: 6735 examples


## 3. Prepare Tokenizer and Model

Let's check our dataset format and clean if necessary:


```python
# Check first few examples in your dataset
print("First example in train_dataset:", train_dataset[0])

# Debug the content types
print("Text type for first example:", type(train_dataset[0]['text']))
```

    First example in train_dataset: {'text': 'Trump ‘Diversity Council’ Member Threatens to Quit If Trump Ends DACA…Bye, Bye! [Video] A member of President Trump s Diversity Council is threatening to quit because he opposes Trump s cancelation of DACA. Bye Bye!Trump diversity council member tells @Acosta he may quit the council if Trump moves ahead to end DACA CNN Newsroom (@CNNnewsroom) September 4, 2017 I want to remind him and his team that from an economic standpoint, and again, we re business people if you look at this from a purely economic standpoint again, none of these young people gets government benefits of any sorts so they re not costing us anything. They pay over $2 billion in taxes Is anyone else out there sick of the American people being told illegals cost nothing?DACA Will Cost Americans And Their Government A Huge Amount of Money.On average, people with college degrees pay more in taxes than they receive in government benefits. People without a degree consume more taxes than they pay to federal, state and local tax officials.In 2013, a Heritage Foundation study showed that amnesty for 11 million illegals would spike federal spending by $6,300 billion over the next five decades. That is roughly equivalent to $550,000 per illegal, or $10,000 per illegal per year, much of which will be spent when the immigrant becomes eligible for Social Security and Medicare. That cost estimate does not include the extra costs created when immigrants use their new legal powers as a citizen to bring in more low-skilled migrants.If those 3 million DACA people and their parents soon become legal residents or citizens, then Obama s DACA will cost Americans roughly $1,700 billion over the next 50 years, according to Heritage Foundation s numbers.Moreover, the DACA migrants add to the flood of illegal labor that has driven down wages for ordinary Americans, including urban youths and recent immigrants. Currently, Americans lose roughly $500 billion a year from their salaries because of the immigration tax caused by cheap labor according to the academies report.Via: GP', 'label': 0}
    Text type for first example: <class 'str'>


Define a cleaning function for the dataset:


```python
# Clean the dataset before tokenization
def clean_dataset(example):
    example['text'] = str(example['text']) if example['text'] is not None else ""
    return example
```

Apply the cleaning function to our datasets:


```python
# Apply cleaning to all datasets
train_dataset = train_dataset.map(clean_dataset)
val_dataset = val_dataset.map(clean_dataset)
test_dataset = test_dataset.map(clean_dataset)
```


    Map:   0%|          | 0/31428 [00:00<?, ? examples/s]



    Map:   0%|          | 0/6735 [00:00<?, ? examples/s]



    Map:   0%|          | 0/6735 [00:00<?, ? examples/s]


Initialize the RoBERTa tokenizer:


```python
# Initialize the RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Define the maximum sequence length
max_length = 512  # This is the maximum that most transformer models can handle
```


    tokenizer_config.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]



    vocab.json:   0%|          | 0.00/899k [00:00<?, ?B/s]



    merges.txt:   0%|          | 0.00/456k [00:00<?, ?B/s]



    tokenizer.json:   0%|          | 0.00/1.36M [00:00<?, ?B/s]



    config.json:   0%|          | 0.00/481 [00:00<?, ?B/s]


Define the tokenization function:


```python
# Function to tokenize the dataset
def tokenize_function(examples):
    # Convert all text entries to strings and handle potential None values
    texts = [str(text) if text is not None else "" for text in examples['text']]
    
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors=None  # Don't return tensors in batch mode
    )
```

Apply tokenization to our datasets:


```python
# Apply tokenization to our datasets
train_tokenized = train_dataset.map(tokenize_function, batched=True)
val_tokenized = val_dataset.map(tokenize_function, batched=True)
test_tokenized = test_dataset.map(tokenize_function, batched=True)
```


    Map:   0%|          | 0/31428 [00:00<?, ? examples/s]



    Map:   0%|          | 0/6735 [00:00<?, ? examples/s]



    Map:   0%|          | 0/6735 [00:00<?, ? examples/s]


Set the format for PyTorch:


```python
# Set the format for PyTorch after tokenization
train_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_tokenized.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
```

## 4. Define Metrics and Evaluation Strategy

Define our evaluation metrics:


```python
# Function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
```

## 5. Initialize Model for Fine-tuning

Initialize the RoBERTa model for sequence classification:


```python
# Initialize the RoBERTa model for sequence classification
model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base',
    num_labels=2  # Binary classification: 0 for fake, 1 for real
)
```

    Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`



    model.safetensors:   0%|          | 0.00/499M [00:00<?, ?B/s]


    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


Move the model to the appropriate device:


```python
# Move model to device (GPU if available)
model.to(device)
```




    RobertaForSequenceClassification(
      (roberta): RobertaModel(
        (embeddings): RobertaEmbeddings(
          (word_embeddings): Embedding(50265, 768, padding_idx=1)
          (position_embeddings): Embedding(514, 768, padding_idx=1)
          (token_type_embeddings): Embedding(1, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (encoder): RobertaEncoder(
          (layer): ModuleList(
            (0-11): 12 x RobertaLayer(
              (attention): RobertaAttention(
                (self): RobertaSdpaSelfAttention(
                  (query): Linear(in_features=768, out_features=768, bias=True)
                  (key): Linear(in_features=768, out_features=768, bias=True)
                  (value): Linear(in_features=768, out_features=768, bias=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
                (output): RobertaSelfOutput(
                  (dense): Linear(in_features=768, out_features=768, bias=True)
                  (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                  (dropout): Dropout(p=0.1, inplace=False)
                )
              )
              (intermediate): RobertaIntermediate(
                (dense): Linear(in_features=768, out_features=3072, bias=True)
                (intermediate_act_fn): GELUActivation()
              )
              (output): RobertaOutput(
                (dense): Linear(in_features=3072, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
          )
        )
      )
      (classifier): RobertaClassificationHead(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (out_proj): Linear(in_features=768, out_features=2, bias=True)
      )
    )



## 6. Define Training Arguments and Trainer

Configure the training parameters:


```python
# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # Output directory for model checkpoints
    num_train_epochs=3,              # Number of training epochs
    per_device_train_batch_size=8,   # Batch size for training
    per_device_eval_batch_size=16,   # Batch size for evaluation
    warmup_steps=500,                # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # Strength of weight decay
    logging_dir='./logs',            # Directory for storing logs
    logging_steps=100,               # Log every X steps
    eval_strategy="epoch",           # Evaluate every epoch
    save_strategy="epoch",           # Save model checkpoint every epoch
    load_best_model_at_end=True,     # Load the best model at the end
    metric_for_best_model="f1",      # Use F1 score to determine the best model
    push_to_hub=False,               # Don't push to Hugging Face Hub
    report_to="none",                # Disable reporting to avoid wandb or other services
    learning_rate=1e-5               # Slightly lower learning rate for RoBERTa
)
```

Create the Trainer:


```python
# Create the Trainer
trainer = Trainer(
    model=model,                         # The instantiated model to train
    args=training_args,                  # Training arguments
    train_dataset=train_tokenized,       # Training dataset
    eval_dataset=val_tokenized,          # Evaluation dataset
    compute_metrics=compute_metrics,     # The function to compute metrics
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Early stopping
)
```

## 7. Fine-tune the Model

Start the timer to measure training time:


```python
# Start the timer to measure training time
start_time = time.time()
```

Train the model:


```python
# Train the model
trainer.train()
```



    <div>

      <progress value='5895' max='5895' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [5895/5895 1:37:45, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Epoch</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>0.000500</td>
      <td>0.000546</td>
      <td>0.999703</td>
      <td>0.999703</td>
      <td>0.999703</td>
      <td>0.999703</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.000000</td>
      <td>0.000010</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.000000</td>
      <td>0.000006</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=5895, training_loss=0.018914131428378906, metrics={'train_runtime': 5867.651, 'train_samples_per_second': 16.068, 'train_steps_per_second': 1.005, 'total_flos': 2.480716274356224e+16, 'train_loss': 0.018914131428378906, 'epoch': 3.0})



Calculate and display the training time:


```python
# Calculate training time
training_time = time.time() - start_time
print(f"Training completed in {training_time/60:.2f} minutes")
```

    Training completed in 97.80 minutes


Save the fine-tuned model:


```python
# Save the fine-tuned model
trainer.save_model("./roberta-fake-news-detector")
```

## 8. Evaluate Model Performance

Evaluate the model on the test set:


```python
# Evaluate the model on the test set
test_results = trainer.evaluate(test_tokenized)
print(f"Test results: {test_results}")
```





    Test results: {'eval_loss': 9.521836545900442e-06, 'eval_accuracy': 1.0, 'eval_f1': 1.0, 'eval_precision': 1.0, 'eval_recall': 1.0, 'eval_runtime': 116.9385, 'eval_samples_per_second': 57.594, 'eval_steps_per_second': 1.804, 'epoch': 3.0}


Get predictions on the test set:


```python
# Get predictions on the test set
test_pred = trainer.predict(test_tokenized)
y_preds = np.argmax(test_pred.predictions, axis=1)
y_true = test_pred.label_ids
```

Create and display the confusion matrix:


```python
# Create confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_preds)
print("Confusion Matrix:")
print(cm)
```

    Confusion Matrix:
    [[3523    0]
     [   0 3212]]


Plot the confusion matrix:


```python
# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('RoBERTa Confusion Matrix')
plt.savefig('roberta_confusion_matrix.png')
plt.show()
```


    
![png](output_55_0.png)
    


Print the classification report:


```python
# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_preds, target_names=['Fake News', 'Real News']))
```

    
    Classification Report:
                  precision    recall  f1-score   support
    
       Fake News       1.00      1.00      1.00      3523
       Real News       1.00      1.00      1.00      3212
    
        accuracy                           1.00      6735
       macro avg       1.00      1.00      1.00      6735
    weighted avg       1.00      1.00      1.00      6735
    


## 9. Analyze Misclassified Examples

Find and count misclassified examples:


```python
# Get indices of misclassified examples
misclassified_indices = np.where(y_preds != y_true)[0]
print(f"Number of misclassified examples: {len(misclassified_indices)}")
```

    Number of misclassified examples: 0


## 10. Model Performance Comparison and Conclusion

Create a comparison table of model performances:


```python
# Create a comparison table of model performances
models = ['TF-IDF + ML', 'DistilBERT', 'TinyBERT', 'RoBERTa']
accuracy = [0.984, 0.9996, 0.9991, test_results['eval_accuracy']]  # Replace with actual value from RoBERTa
f1_scores = [0.984, 0.9996, 0.9991, test_results['eval_f1']]  # Replace with actual value from RoBERTa
training_times = ['N/A', '48.69 minutes', '8.99 minutes', f"{training_time/60:.2f} minutes"]

comparison_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'F1 Score': f1_scores,
    'Training Time': training_times
})

print("Model Performance Comparison:")
display(comparison_df)
```

    Model Performance Comparison:



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>F1 Score</th>
      <th>Training Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TF-IDF + ML</td>
      <td>0.9840</td>
      <td>0.9840</td>
      <td>N/A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DistilBERT</td>
      <td>0.9996</td>
      <td>0.9996</td>
      <td>48.69 minutes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TinyBERT</td>
      <td>0.9991</td>
      <td>0.9991</td>
      <td>8.99 minutes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RoBERTa</td>
      <td>1.0000</td>
      <td>1.0000</td>
      <td>97.80 minutes</td>
    </tr>
  </tbody>
</table>
</div>


In this notebook, I've fine-tuned a RoBERTa model for fake news detection on the ISOT dataset. Here are the key findings:

1. **Performance Comparison**: RoBERTa achieves excellent accuracy, comparable to DistilBERT and TinyBERT. The performance metrics confirm that transformer-based models are highly effective for fake news detection.

2. **Training Efficiency**: Despite being a larger model than TinyBERT, RoBERTa offers a good balance between performance and computational requirements.

3. **Error Analysis**: The misclassified examples provide insights into challenging cases that can guide further improvements in the model or data preprocessing.

4. **Generalization Potential**: RoBERTa's more robust pretraining may provide better generalization to unseen fake news examples, especially those that differ significantly from the training data.

## Next Steps

1. **Ensemble Methods**: Combine predictions from multiple models (RoBERTa, DistilBERT, TinyBERT) to potentially improve overall performance and robustness.

2. **Additional Training Data**: Incorporate different fake news datasets to improve generalization across different topics and styles.

3. **Model Distillation**: Distill knowledge from RoBERTa into smaller models like TinyBERT to improve their performance while maintaining efficiency.

4. **Explainability**: Apply techniques like LIME or SHAP to understand which parts of the text are most influential in the model's decisions.

5. **Adversarial Testing**: Create challenging examples that attempt to fool the model, which can help identify weaknesses and guide improvements.

RoBERTa offers a powerful approach to fake news detection, complementing our previous work with DistilBERT and TinyBERT. While all three transformer models achieve excellent performance on the ISOT dataset, they each have different characteristics that may be beneficial in different deployment scenarios.
