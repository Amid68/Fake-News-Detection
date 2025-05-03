# DistilBERT Fake News Detector - Project Notes

## What I Did

I built a lightweight fake news detector using DistilBERT that can analyze news articles and determine if they're likely real or fake. The project used the FakeNewsNet dataset with about 45,000 labeled articles (approximately 23,500 fake and 21,500 real).

## The Training Process

1. **Data Preparation**:
   - Combined two datasets (fake and real news)
   - Preprocessed text (lowercase, removed URLs, cleaned whitespace)
   - Created combined features from titles and article content
   - Split data into train/validation/test sets (68/12/20%)

2. **Model Setup**:
   - Used DistilBERT (smaller version of BERT with ~67M parameters)
   - Created custom PyTorch datasets and dataloaders
   - Set up training with a max sequence length of 512 tokens
   - Used batch size of 8 to fit in memory

3. **Training Configuration**:
   - Used Hugging Face's Trainer API
   - Applied class weights to handle slight class imbalance
   - Trained for 3 epochs with learning rate warmup
   - Used evaluation steps during training to monitor progress
   - Added early stopping based on F1 score

## Problems I Faced

The biggest issue was that my initial model kept classifying almost everything as fake news. This happened because:

1. **Class Imbalance**: The dataset had slightly more fake articles (52%) than real ones (48%)
2. **Threshold Problem**: Using a fixed 0.5 threshold wasn't optimal
3. **Overfitting**: The model was memorizing patterns instead of learning general features
4. **Training Bias**: The model was biased toward the majority class

## How I Fixed the Issues

1. **Applied Class Weights**: 
   - Calculated balanced weights for each class
   - Added a custom loss function that used these weights

2. **Threshold Calibration**:
   - Found the optimal classification threshold using precision-recall curves
   - This helped balance precision and recall instead of using the default 0.5

3. **Evaluation Strategy**:
   - Added frequent evaluation during training
   - Used early stopping to prevent overfitting

4. **Improved Inference Function**:
   - Created a Django-ready function with threshold parameter
   - Made outputs more detailed with confidence scores for both classes
   - Added explicit real/fake probability scores

## Results

The final model achieved:
- 99.8% accuracy on the test set
- Excellent F1, precision, and recall scores
- Super fast inference (about 8ms per article)
- Low memory usage (under 2MB during inference)

However, I still need to test it more with real-world examples. Current results suggested it might be overconfident on some legitimate news that contains unusual or sensational (but true) content.

## Next Steps

If I continue with this project, I should:
1. Try smaller models (TinyBERT) for even faster inference
2. Get more diverse training data including recent news
3. Add explainability features to understand why articles are classified certain ways
4. Implement regular re-training as news patterns change over time

## Lessons Learned

1. Always examine class balance and apply appropriate weights
2. Don't use default classification thresholds without testing
3. Monitor training closely to catch issues early
4. Test with real-world examples, not just test sets

This project showed me that while building a fake news detector is technically feasible, it requires careful tuning to avoid bias and ensure practical usefulness.