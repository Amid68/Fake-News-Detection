#!/usr/bin/env python3
"""
Verify all model parameters and sizes for efficiency table
"""

import pickle
import os
from transformers import AutoModel, AutoConfig

def get_file_size_mb(filepath):
    """Get file size in MB"""
    return os.path.getsize(filepath) / (1024 * 1024)

def get_transformer_metrics(model_path, model_name):
    """Get metrics for transformer models"""
    try:
        # Load model
        model = AutoModel.from_pretrained(model_path)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Get model file size (safetensors)
        safetensors_path = os.path.join(model_path, "model.safetensors")
        model_size_mb = get_file_size_mb(safetensors_path)
        
        print(f"\n=== {model_name.upper()} ===")
        print(f"Parameters: {total_params:,}")
        print(f"Model size: {model_size_mb:.2f} MB")
        
        return total_params, model_size_mb
        
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None, None

def get_baseline_metrics():
    """Get metrics for baseline models"""
    
    print("=== LOGISTIC REGRESSION ===")
    try:
        with open('ml_models/baseline/lr_model.pkl', 'rb') as f:
            lr_model = pickle.load(f)
        
        param_count = lr_model.coef_.size + 1  # coefficients + bias
        file_size_mb = get_file_size_mb('ml_models/baseline/lr_model.pkl')
        
        print(f"Parameters: {param_count:,}")
        print(f"Model size: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== RANDOM FOREST ===")
    try:
        with open('ml_models/baseline/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        
        total_nodes = sum(tree.tree_.node_count for tree in rf_model.estimators_)
        file_size_mb = get_file_size_mb('ml_models/baseline/rf_model.pkl')
        
        print(f"Total nodes: {total_nodes:,} (recommend using N/A in table)")
        print(f"Model size: {file_size_mb:.2f} MB")
        print(f"Trees: {rf_model.n_estimators}")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    print("VERIFYING ALL MODEL METRICS")
    print("=" * 50)
    
    # Baseline models
    get_baseline_metrics()
    
    # Transformer models
    transformer_models = [
        ('ml_models/distilbert_welfake_model', 'DistilBERT'),
        ('ml_models/albert_welfake_model', 'ALBERT'),
        ('ml_models/mobilebert_welfake_model', 'MobileBERT'),
        ('ml_models/tinybert_welfake_model', 'TinyBERT')
    ]
    
    results = {}
    for model_path, model_name in transformer_models:
        params, size = get_transformer_metrics(model_path, model_name)
        if params:
            results[model_name] = {'params': params, 'size': size}
    
    # Summary table format
    print("\n" + "=" * 50)
    print("SUMMARY FOR YOUR TABLE:")
    print("=" * 50)
    print("Model               | Parameters    | Size (MB)")
    print("-" * 50)
    print("Logistic Regression | (see above)   | (see above)")
    print("Random Forest       | N/A           | (see above)")
    
    for name, data in results.items():
        print(f"{name:<18} | {data['params']:>12,} | {data['size']:>8.2f}")

if __name__ == "__main__":
    main()
