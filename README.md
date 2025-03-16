# Natural Language Processing: Disaster Tweet Classification

**Aim**: Build and compare text classification models using traditional ML, neural networks, and fine-tuned transformers for disaster tweet classification.  
**Dataset**: [NLP with Disaster Tweets (Kaggle)](https://www.kaggle.com/competitions/nlp-getting-started)  

---

## Assignment Requirements Overview
The assignment involves three main tasks:  
1. **Traditional ML Models**: TF-IDF and Bag-of-Words with classifiers like Logistic Regression and SVM.  
2. **Neural Networks**: Build and train RNN, LSTM, GRU, and CNN models from scratch.  
3. **Fine-Tuned Transformers**: Adapt pre-trained models (DistilBERT, ELECTRA) for the task.  

Each task includes:  
- Preprocessing (tokenization, lemmatization, etc.)  
- Model training and hyperparameter tuning  
- Kaggle submission and performance evaluation  

---

## Approach

### 1. Preprocessing Pipeline
#### Tokenization & Lemmatization
- **Tokenizer**: NLTK `word_tokenize`  
  - Chosen for robust handling of Twitter-specific syntax (hashtags, mentions).  
- **Lemmatizer**: WordNetLemmatizer  
  - Preferred over stemming to preserve semantic meaning (e.g., "running" → "run").  

#### Key Preprocessing Steps:
1. **Text Cleaning**:  
   - Remove URLs, mentions, and special characters using regex.  
   - Convert text to lowercase and strip leading/trailing spaces.  
2. **Contraction Handling**:  
   - Expand contractions (e.g., "can't" → "cannot") using the `contractions` library.  
3. **Stopword Removal**:  
   - Use NLTK's English stopwords with custom additions (`http`, `https`, `rt`, etc.).  
4. **Token Filtering**:  
   - Remove single-character tokens and lemmatize remaining words.  

### 2. Model Architectures
#### Task 1: Traditional Models
| Model              | Rationale                                                                 | Hyperparameters Tuned                     |
|--------------------|---------------------------------------------------------------------------|-------------------------------------------|
| TF-IDF + SVM       | Effective for high-dimensional sparse data                                | `C`, `ngram_range`, `max_features`       |
| BoW + Naive Bayes  | Probabilistic baseline for text classification                            | `alpha`, `fit_prior`                     |
| Gradient Boosting  | Benchmark against ensemble methods (despite sparse data challenges)       | `n_estimators`, `max_depth`, `learning_rate` |

#### Task 2: Neural Networks
| Architecture       | Design Choices                                                                 |  
|--------------------|-------------------------------------------------------------------------------|  
| **Bidirectional LSTM** | Captures contextual dependencies in both text directions                     |  
| **CNN**            | 1D convolutions with `kernel_size=3` to detect local n-gram patterns          |  
| **GRU**            | Balance between LSTM complexity and RNN speed                                |  

**Embedding Layer**:  
- Trained from scratch (`embed_dim=64-128`) due to Twitter-specific vocabulary.  

#### Task 3: Pre-trained Models
| Model       | Selection Rationale                                  | Fine-Tuning Strategy                     |  
|-------------|------------------------------------------------------|------------------------------------------|  
| **DistilBERT** | 40% smaller than BERT with 95% performance          | Layer-wise learning rate decay           |  
| **ELECTRA**    | Efficient pretraining via replaced token detection   | Frozen embeddings + tuned classification head |  

---

## Repository Structure
```
├── Dataset/  
│   ├── train.csv              # Training data (7,613 tweets)  
│   ├── test.csv               # Test data (3,263 tweets)  
│   └── sample_submission.csv  # Submission format  

├── Traditional Models/  
│   ├── models.ipynb           # TF-IDF/BoW implementations  
│   ├── bow_gb_submission.csv  # Gradient Boosting predictions  
│   ├── bow_lr_submission.csv  # Logistic Regression predictions  
│   ├── bow_mnb_submission.csv # Naive Bayes predictions  
│   ├── bow_svm_submission.csv # SVM predictions  
│   ├── tfidf_gb_submission.csv# Gradient Boosting predictions  
│   ├── tfidf_lr_submission.csv# Logistic Regression predictions  
│   ├── tfidf_mnb_submission.csv# Naive Bayes predictions  
│   └── tfidf_svm_submission.csv# SVM predictions  

├── Neural Network Models/  
│   ├── models.ipynb           # RNN/LSTM/CNN implementations  
│   ├── CNN_submission.csv     # CNN predictions  
│   ├── GRU_submission.csv     # GRU predictions  
│   ├── LSTM_submission.csv    # LSTM predictions  
│   └── RNN_submission.csv     # RNN predictions  

├── Tuned Existing Models/  
│   ├── models.ipynb           # Transformer fine-tuning code  
│   ├── distilbert_submission.csv # DistilBERT predictions  
│   ├── mobilebert_submission.csv # MobileBERT predictions  
│   └── electra_submission.csv    # ELECTRA predictions  
```

---

## Results & Compliance with Requirements
### Hyperparameter Tuning
- **Traditional Models**: GridSearchCV with 5-fold stratified validation.  
- **Neural Networks**: Keras Tuner for embedding dimensions, dropout rates, and layer sizes.  
- **Transformers**: Learning rate sweeps (2e-5 to 5e-5) and batch size optimization.  

### Kaggle Performance
| Model Type         | Best Val F1 | Kaggle F1* | Resource Usage |  
|--------------------|-------------|------------|----------------|  
| TF-IDF + SVM       | 0.778       | 0.752      | Low (CPU)      |  
| CNN (scratch)      | 0.776       | 0.741      | Medium (GPU)   |  
| **DistilBERT**     | **0.816**   | **0.789**  | High (GPU)     |  

_Kaggle scores based on submissions_

---

## Conclusions (Assignment Requirement Coverage)
### 1. Best Model
- **DistilBERT** achieved the highest F1-score (0.816) but required GPU resources.  
- **TF-IDF + SVM** provided the best CPU performance (0.778 F1) and is suitable for resource-constrained environments.  

### 2. Improvement Strategies
- **Class Imbalance**: Use weighted loss functions or oversampling techniques like SMOTE.  
- **Feature Engineering**: Incorporate additional text features (e.g., emojis, sentiment scores).  
- **Ensemble Methods**: Combine predictions from top-performing models (e.g., TF-IDF SVM + DistilBERT).  

### 3. Challenges Faced
- **Preprocessing Complexity**: Balancing thorough cleaning with retaining useful information (e.g., hashtags).  
- **Hardware Limitations**: Transformer fine-tuning required GPU acceleration for efficient training.  

---

## Usage
1. **Traditional Models**  
```bash
jupyter notebook "Traditional Models/models.ipynb"
```

2. **Neural Networks**  
```bash
jupyter notebook "Neural Network Models/models.ipynb"
```

3. **Transformers**  
```bash
jupyter notebook "Tuned Existing Models/models.ipynb"
```

---

## References
1. [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
2. [Hugging Face Transformers](https://huggingface.co/docs/transformers/)  
3. [Kaggle Competition Forum](https://www.kaggle.com/competitions/nlp-getting-started/discussion)  
