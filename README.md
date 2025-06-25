# Fake News Detection: Comparative Analysis of ML Models

A comprehensive comparison of traditional machine learning and transformer-based models for fake news detection, developed as a Bachelor's graduation project at An-Najah National University.

## Overview

This project evaluates and compares various machine learning approaches for detecting fake news, from traditional methods to state-of-the-art transformer models. The study provides practical insights into the trade-offs between accuracy, computational efficiency, and generalization capabilities.

## Models Compared

### Traditional ML Models
- **Logistic Regression** with TF-IDF features
- **Random Forest** with TF-IDF features

### Transformer Models
- **DistilBERT** - Distilled BERT with 40% fewer parameters
- **ALBERT** - Parameter-sharing BERT variant
- **MobileBERT** - Mobile-optimized BERT
- **TinyBERT** - Knowledge-distilled compact BERT

## Dataset

- **Primary**: WELFake dataset (71,537 cleaned articles)
- **External Evaluation**: Manual real news + AI-generated fake news
- **Classes**: Binary classification (Real vs Fake news)
- **Balance**: ~50% real, ~50% fake news

## Key Findings

### Performance (WELFake Test Set)
1. **ALBERT**: 99.66% accuracy
2. **DistilBERT**: 99.65% accuracy  
3. **MobileBERT**: 99.61% accuracy
4. **TinyBERT**: 99.28% accuracy
5. **Logistic Regression**: 95.96% accuracy
6. **Random Forest**: 95.61% accuracy

### Generalization (External Datasets)
- **Random Forest**: Best generalization (97.79% accuracy)
- **Logistic Regression**: Strong cross-domain performance
- **Transformer models**: Significant performance drops on new data

### Efficiency Highlights
- **MobileBERT**: Fastest transformer inference (8.97ms/sample)
- **ALBERT**: Smallest model size (44.58 MB)
- **TinyBERT**: Fastest training time (14.87 minutes)
- **Logistic Regression**: Ultra-fast inference (0.463ms/sample)

