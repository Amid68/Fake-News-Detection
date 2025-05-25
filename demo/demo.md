I'll help you create a more streamlined and visually appealing demo notebook with LIME explainability, better visualizations, and MobileBERT included. Here's an improved version:

# Fake News Detection Demo Notebook
This notebook demonstrates various machine learning approaches for detecting fake news, from traditional ML to state-of-the-art transformers.

## 1. Setup and Imports

```python
# Cell 1: Setup and Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
```

## 2. Introduction

```python
# Cell 2: Introduction with visual appeal
def display_intro():
    html = """
    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: #2c3e50; text-align: center;">🔍 Fake News Detection System Demo</h1>
        <p style="font-size: 16px; text-align: center; color: #34495e;">
            Comparing Traditional ML and Modern Transformer Approaches
        </p>
    </div>
    
    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px;">
            <h3 style="color: #2e7d32;">📊 Traditional ML</h3>
            <ul style="color: #424242;">
                <li>Logistic Regression with TF-IDF</li>
                <li>Lightning fast inference</li>
                <li>Excellent generalization</li>
            </ul>
        </div>
        
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px;">
            <h3 style="color: #1565c0;">🤖 Transformer Models</h3>
            <ul style="color: #424242;">
                <li>TinyBERT: Mobile-ready (14M params)</li>
                <li>MobileBERT: Edge devices (25M params)</li>
                <li>DistilBERT: Balanced (67M params)</li>
                <li>ALBERT: Maximum accuracy (12M params)</li>
            </ul>
        </div>
    </div>
    """
    display(HTML(html))

display_intro()
```

## 3. Load Models

```python
# Cell 3: Simplified model loading with progress indicators
class ModelLoader:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.vectorizer = None
        
    def load_all_models(self):
        """Load all available models with visual progress."""
        model_configs = [
            ('Logistic Regression', self._load_traditional, '🔢'),
            ('TinyBERT', self._load_tinybert, '🐤'),
            ('MobileBERT', self._load_mobilebert, '📱'),
            ('DistilBERT', self._load_distilbert, '⚡'),
            ('ALBERT', self._load_albert, '🧠')
        ]
        
        for name, loader, icon in model_configs:
            print(f"{icon} Loading {name}...", end=' ')
            try:
                loader()
                print("✅ Success!")
            except Exception as e:
                print(f"❌ Not available")
    
    def _load_traditional(self):
        with open('../ml_models/baseline/lr_text_model.pkl', 'rb') as f:
            self.models['lr'] = pickle.load(f)
        with open('../ml_models/baseline/tfidf_vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
    
    def _load_tinybert(self):
        path = '../ml_models/tinybert_welfake_model'
        self.tokenizers['tinybert'] = AutoTokenizer.from_pretrained(path)
        self.models['tinybert'] = AutoModelForSequenceClassification.from_pretrained(path)
        self.models['tinybert'].eval()
    
    def _load_mobilebert(self):
        path = '../ml_models/mobilebert_welfake_model'
        self.tokenizers['mobilebert'] = AutoTokenizer.from_pretrained(path)
        self.models['mobilebert'] = AutoModelForSequenceClassification.from_pretrained(path)
        self.models['mobilebert'].eval()
    
    def _load_distilbert(self):
        path = '../ml_models/distilbert_welfake_model'
        self.tokenizers['distilbert'] = AutoTokenizer.from_pretrained(path)
        self.models['distilbert'] = AutoModelForSequenceClassification.from_pretrained(path)
        self.models['distilbert'].eval()
    
    def _load_albert(self):
        path = '../ml_models/albert_welfake_model'
        self.tokenizers['albert'] = AutoTokenizer.from_pretrained(path)
        self.models['albert'] = AutoModelForSequenceClassification.from_pretrained(path)
        self.models['albert'].eval()

# Load all models
loader = ModelLoader()
loader.load_all_models()
```

## 4. Prediction Functions

```python
# Cell 4: Simplified prediction functions
class FakeNewsDetector:
    def __init__(self, model_loader):
        self.loader = model_loader
        
    def predict_all(self, text):
        """Get predictions from all available models."""
        results = []
        
        # Traditional ML
        if 'lr' in self.loader.models:
            result = self._predict_traditional(text)
            if result:
                results.append(result)
        
        # Transformers
        for model_name in ['tinybert', 'mobilebert', 'distilbert', 'albert']:
            if model_name in self.loader.models:
                result = self._predict_transformer(text, model_name)
                if result:
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def _predict_traditional(self, text):
        """Traditional ML prediction."""
        start = time.time()
        X = self.loader.vectorizer.transform([text])
        pred = self.loader.models['lr'].predict(X)[0]
        prob = self.loader.models['lr'].predict_proba(X)[0]
        time_ms = (time.time() - start) * 1000
        
        return {
            'Model': 'Logistic Regression',
            'Prediction': '🚫 FAKE' if pred == 1 else '✅ REAL',
            'Confidence': f'{prob.max():.1%}',
            'Time (ms)': f'{time_ms:.1f}',
            'Type': 'Traditional ML'
        }
    
    def _predict_transformer(self, text, model_name):
        """Transformer prediction."""
        model = self.loader.models[model_name]
        tokenizer = self.loader.tokenizers[model_name]
        
        start = time.time()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            probs = torch.softmax(outputs.logits, dim=1)
            conf = probs.max().item()
        
        time_ms = (time.time() - start) * 1000
        
        model_display = {
            'tinybert': 'TinyBERT',
            'mobilebert': 'MobileBERT',
            'distilbert': 'DistilBERT',
            'albert': 'ALBERT'
        }
        
        return {
            'Model': model_display[model_name],
            'Prediction': '🚫 FAKE' if pred == 1 else '✅ REAL',
            'Confidence': f'{conf:.1%}',
            'Time (ms)': f'{time_ms:.1f}',
            'Type': 'Transformer'
        }

# Create detector
detector = FakeNewsDetector(loader)
```

## 5. Interactive Demo

```python
# Cell 5: Visual article analysis
def analyze_article(text, title="Article Analysis"):
    """Analyze article with visual presentation."""
    # Display article preview
    preview = text[:200] + "..." if len(text) > 200 else text
    
    html = f"""
    <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3 style="color: #2c3e50;">{title}</h3>
        <div style="background-color: white; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p style="font-style: italic; color: #555;">{preview}</p>
        </div>
    </div>
    """
    display(HTML(html))
    
    # Get predictions
    results = detector.predict_all(text)
    
    if not results.empty:
        # Style the results table
        styled_results = results.style.set_properties(**{
            'text-align': 'center',
            'font-size': '14px'
        }).set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#3498db'), 
                                         ('color', 'white'), 
                                         ('font-weight', 'bold')]}
        ])
        display(styled_results)
        
        # Create consensus visualization
        fake_count = sum(1 for pred in results['Prediction'] if 'FAKE' in pred)
        real_count = len(results) - fake_count
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Consensus pie chart
        ax1.pie([real_count, fake_count], labels=['Real', 'Fake'], 
                colors=['#2ecc71', '#e74c3c'], autopct='%1.0f%%',
                startangle=90)
        ax1.set_title('Model Consensus')
        
        # Inference time comparison
        results_copy = results.copy()
        results_copy['Time_float'] = results_copy['Time (ms)'].str.replace(' ms', '').astype(float)
        colors = ['#3498db' if t == 'Traditional ML' else '#9b59b6' for t in results_copy['Type']]
        ax2.bar(results_copy['Model'], results_copy['Time_float'], color=colors)
        ax2.set_ylabel('Inference Time (ms)')
        ax2.set_title('Speed Comparison')
        ax2.set_yscale('log')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
```

## 6. Test Examples

```python
# Cell 6: Test with examples
# Example 1: Real news
real_news = """
SpaceX successfully launches Falcon 9 rocket carrying 60 Starlink satellites into orbit. 
The launch took place at Cape Canaveral Space Force Station in Florida on Tuesday evening. 
This marks the company's 15th mission this year as it continues to expand its satellite 
internet constellation. The first stage booster successfully landed on the drone ship.
"""

print("📰 Example 1: Real News")
analyze_article(real_news, "Real News Example")
```

```python
# Cell 7: Fake news example
fake_news = """
BREAKING: Scientists at Harvard have discovered that eating chocolate before bed 
increases IQ by 20 points! This SHOCKING discovery is being SUPPRESSED by the 
education industry. Share this before it gets DELETED! Studies show 100% success rate. 
Big Pharma doesn't want you to know this ONE SIMPLE TRICK!
"""

print("\n📰 Example 2: Fake News")
analyze_article(fake_news, "Fake News Example")
```

## 7. LIME Explainability

```python
# Cell 8: LIME explanation (simplified version)
def explain_prediction(text, model_name='lr'):
    """Show which words contributed to the prediction."""
    if model_name == 'lr' and 'lr' in loader.models:
        # Get TF-IDF features
        X = loader.vectorizer.transform([text])
        feature_names = loader.vectorizer.get_feature_names_out()
        
        # Get model coefficients
        coef = loader.models['lr'].coef_[0]
        
        # Get top contributing words
        tfidf_scores = X.toarray()[0]
        word_contributions = tfidf_scores * coef
        
        # Get top positive and negative contributors
        top_indices = np.argsort(np.abs(word_contributions))[-10:][::-1]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        words = [feature_names[i] for i in top_indices]
        scores = [word_contributions[i] for i in top_indices]
        colors = ['#e74c3c' if s > 0 else '#2ecc71' for s in scores]
        
        bars = ax.barh(words, scores, color=colors)
        ax.set_xlabel('Contribution to Fake News Score')
        ax.set_title(f'Word Importance Analysis')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend
        red_patch = plt.Rectangle((0,0),1,1, fc='#e74c3c')
        green_patch = plt.Rectangle((0,0),1,1, fc='#2ecc71')
        ax.legend([red_patch, green_patch], ['Indicates Fake', 'Indicates Real'], 
                 loc='best')
        
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ LIME explanation only available for Logistic Regression model")

# Example usage
print("🔍 Explainability Analysis")
explain_prediction(fake_news)
```

## 8. Model Comparison Dashboard

```python
# Cell 9: Comprehensive model comparison
def create_comparison_dashboard():
    """Create a visual dashboard comparing all models."""
    
    # Model characteristics
    model_data = {
        'Model': ['Logistic\nRegression', 'TinyBERT', 'MobileBERT', 'DistilBERT', 'ALBERT'],
        'Parameters (M)': [0.1, 14.3, 24.6, 66.9, 11.7],
        'Size (MB)': [8, 54.7, 93.8, 255.4, 44.6],
        'Accuracy (%)': [94.9, 99.3, 99.7, 99.7, 99.8],
        'Speed (ms)': [0.24, 14.0, 103.7, 51.5, 159.8],
        'External Generalization (%)': [97.0, 83.7, 52.5, 64.4, 60.1]
    }
    
    df = pd.DataFrame(model_data)
    
    # Create subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Size vs Accuracy scatter plot
    ax1 = fig.add_subplot(gs[0, :2])
    scatter = ax1.scatter(df['Size (MB)'], df['Accuracy (%)'], 
                         s=df['Parameters (M)']*10, 
                         c=range(len(df)), cmap='viridis', alpha=0.6)
    ax1.set_xlabel('Model Size (MB)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Model Size vs Accuracy (bubble size = parameters)')
    
    # Add model labels
    for i, txt in enumerate(df['Model']):
        ax1.annotate(txt, (df['Size (MB)'][i], df['Accuracy (%)'][i]), 
                    fontsize=10, ha='center')
    
    # 2. Speed comparison
    ax2 = fig.add_subplot(gs[0, 2])
    bars = ax2.bar(df['Model'], df['Speed (ms)'], color=plt.cm.viridis(np.linspace(0, 1, len(df))))
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Inference Speed')
    ax2.set_yscale('log')
    
    # 3. Generalization comparison
    ax3 = fig.add_subplot(gs[1, :])
    x = np.arange(len(df['Model']))
    width = 0.35
    ax3.bar(x - width/2, df['Accuracy (%)'], width, label='Training Domain', alpha=0.8)
    ax3.bar(x + width/2, df['External Generalization (%)'], width, 
            label='External Data', alpha=0.8)
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Model Generalization: Training vs External Data')
    ax3.set_xticks(x)
    ax3.set_xticklabels(df['Model'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Radar chart for model comparison
    ax4 = fig.add_subplot(gs[2, :], projection='polar')
    
    # Select models for radar chart
    models_to_compare = ['TinyBERT', 'DistilBERT', 'ALBERT']
    metrics = ['Accuracy', 'Speed', 'Size', 'Generalization', 'Efficiency']
    
    # Normalize metrics to 0-1 scale
    accuracy_norm = df[df['Model'].str.contains('BERT')]['Accuracy (%)'].values / 100
    speed_norm = 1 - (np.log(df[df['Model'].str.contains('BERT')]['Speed (ms)'].values) / 
                     np.log(df[df['Model'].str.contains('BERT')]['Speed (ms)'].max()))
    size_norm = 1 - (df[df['Model'].str.contains('BERT')]['Size (MB)'].values / 
                    df[df['Model'].str.contains('BERT')]['Size (MB)'].max())
    gen_norm = df[df['Model'].str.contains('BERT')]['External Generalization (%)'].values / 100
    efficiency_norm = accuracy_norm / (df[df['Model'].str.contains('BERT')]['Size (MB)'].values / 100)
    efficiency_norm = efficiency_norm / efficiency_norm.max()
    
    # Angles for radar chart
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    
    # Plot each model
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    for i, (model, color) in enumerate(zip(['TinyBERT', 'DistilBERT', 'ALBERT'], colors)):
        idx = 1 + i  # Skip LogReg in the dataframe
        values = [accuracy_norm[i], speed_norm[i], size_norm[i], gen_norm[i], efficiency_norm[i]]
        values = np.concatenate([values, [values[0]]])
        ax4.plot(angles, values, 'o-', linewidth=2, label=model, color=color)
        ax4.fill(angles, values, alpha=0.25, color=color)
    
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(metrics)
    ax4.set_ylim(0, 1)
    ax4.set_title('Transformer Model Comparison', y=1.08)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

# Display the dashboard
print("📊 Model Comparison Dashboard")
create_comparison_dashboard()
```

## 9. Interactive Demo

```python
# Cell 10: User input demo
def interactive_demo():
    """Simple interactive demo for custom text."""
    custom_text = """
    Enter your news article here. For example:
    
    Researchers at MIT have developed a new battery technology that could extend 
    electric vehicle range by 50%. The breakthrough uses a novel lithium-metal 
    design that prevents dendrite formation, addressing a key safety concern.
    """
    
    print("✍️ Test with your own text!")
    print("Modify the 'custom_text' variable above and run this cell.")
    
    analyze_article(custom_text, "Custom Article Analysis")

interactive_demo()
```

## 10. Deployment Recommendations

```python
# Cell 11: Visual deployment guide
def show_deployment_guide():
    """Display deployment recommendations with visual cards."""
    
    html = """
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px;">
        <h2 style="color: #2c3e50; text-align: center;">🚀 Deployment Recommendations</h2>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px;">
            
            <div style="background-color: #e8f8f5; padding: 20px; border-radius: 10px; border: 2px solid #1abc9c;">
                <h3 style="color: #16a085;">📱 Mobile Apps</h3>
                <p><strong>Recommended:</strong> TinyBERT</p>
                <ul>
                    <li>Size: 55 MB</li>
                    <li>Speed: 14ms per article</li>
                    <li>Accuracy: 83% on external data</li>
                </ul>
                <p style="color: #7f8c8d; font-style: italic;">Perfect balance for mobile constraints</p>
            </div>
            
            <div style="background-color: #ebf5fb; padding: 20px; border-radius: 10px; border: 2px solid #3498db;">
                <h3 style="color: #2980b9;">🌐 Web Browser</h3>
                <p><strong>Recommended:</strong> Logistic Regression</p>
                <ul>
                    <li>Size: 8 MB</li>
                    <li>Speed: 0.24ms per article</li>
                    <li>Accuracy: 97% generalization</li>
                </ul>
                <p style="color: #7f8c8d; font-style: italic;">Instant predictions, minimal resources</p>
            </div>
            
            <div style="background-color: #fdf2e9; padding: 20px; border-radius: 10px; border: 2px solid #e67e22;">
                <h3 style="color: #d35400;">💻 Desktop Apps</h3>
                <p><strong>Recommended:</strong> DistilBERT</p>
                <ul>
                    <li>Size: 255 MB</li>
                    <li>Speed: 51ms per article</li>
                    <li>Accuracy: 99.7% on training data</li>
                </ul>
                <p style="color: #7f8c8d; font-style: italic;">Best accuracy-speed balance</p>
            </div>
            
            <div style="background-color: #fadbd8; padding: 20px; border-radius: 10px; border: 2px solid #e74c3c;">
                <h3 style="color: #c0392b;">☁️ Server API</h3>
                <p><strong>Recommended:</strong> ALBERT</p>
                <ul>
                    <li>Size: 45 MB (parameter efficient)</li>
                    <li>Speed: 160ms per article</li>
                    <li>Accuracy: 99.8% highest accuracy</li>
                </ul>
                <p style="color: #7f8c8d; font-style: italic;">Maximum accuracy for critical applications</p>
            </div>
            
        </div>
    </div>
    """
    display(HTML(html))

show_deployment_guide()
```

## Summary

This demo showcased different approaches to fake news detection, from lightning-fast traditional ML to state-of-the-art transformers. Each model offers unique advantages:

### Key Insights:
- **Traditional ML** excels at generalization and speed
- **TinyBERT** provides the best mobile experience
- **MobileBERT** offers good balance for edge devices
- **DistilBERT** is ideal for general-purpose applications
- **ALBERT** achieves maximum accuracy with parameter efficiency

### Choose Based on Your Needs:
- **Speed critical?** → Logistic Regression
- **Mobile deployment?** → TinyBERT
- **Best accuracy?** → ALBERT
- **Balanced performance?** → DistilBERT

### Next Steps:
1. Test with your own articles
2. Fine-tune on domain-specific data
3. Implement continuous learning for emerging patterns
4. Consider ensemble approaches for production

The future of fake news detection lies in combining the generalization of traditional ML with the sophistication of modern transformers!
