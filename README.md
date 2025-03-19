# Natural Language Processing: Disaster Tweet Classification

**Aim**: Build and compare text classification models using traditional ML, neural networks, and fine-tuned transformers for disaster tweet classification.  
**Dataset**: [NLP with Disaster Tweets (Kaggle)](https://www.kaggle.com/competitions/nlp-getting-started)  

---

## Overview
The involves three main tasks:  
1. **Traditional ML Models**: TF-IDF and Bag-of-Words with classifiers like Logistic Regression and SVM.  
2. **Neural Networks**: Build and train RNN, LSTM, GRU, and CNN models from scratch.  
3. **Fine-Tuned Transformers**: Adapt pre-trained models (DistilBERT, MobileBERT, ELECTRA) for the task.  

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
| **MobileBERT** | 4x faster than BERT, inverted bottleneck            | Layer-wise learning rate decay           |  
| **ELECTRA**    | Efficient pretraining via replaced token detection   | Frozen embeddings + tuned classification head |  

---

## Repository Structure
```
├── Dataset/  
│   ├── train.csv                
│   ├── test.csv                 
│   └── sample_submission.csv  

├── Traditional Models/  
│   ├── models.ipynb            
│   ├── bow_gb_submission.csv    
│   ├── bow_lr_submission.csv   
│   ├── bow_mnb_submission.csv 
│   ├── bow_svm_submission.csv  
│   ├── tfidf_gb_submission.csv  
│   ├── tfidf_lr_submission.csv
│   ├── tfidf_mnb_submission.csv 
│   └── tfidf_svm_submission.csv 

├── Neural Network Models/  
│   ├── models.ipynb           
│   ├── CNN_submission.csv     
│   ├── GRU_submission.csv      
│   ├── LSTM_submission.csv    
│   └── RNN_submission.csv      

├── Tuned Existing Models/  
│   ├── models.ipynb             
│   ├── distilbert_submission.csv 
│   ├── mobilebert_submission.csv   
│   └── electra_submission.csv      
```

---

## Results
### Hyperparameter Tuning
- **Traditional Models**: GridSearchCV with 5-fold stratified validation.  
- **Neural Networks**: Keras Tuner for embedding dimensions, dropout rates, and layer sizes.  
- **Transformers**: Learning rate sweeps (2e-5 to 5e-5) and batch size optimization.  

### Kaggle Performance
| Model Type         | Best Val F1 | Kaggle F1* | Resource Usage |  
|--------------------|-------------|------------|----------------|  
| TF-IDF + SVM       | 0.7783       | 0.7971      | Low           |  
| CNN (scratch)      | 0.7782       | 0.7631     | Medium          |  
| **DistilBERT**     | **0.816**   | **0.8235**  | High          |  

---

## Conclusions
### 1. Best Model
- **DistilBERT** achieved the highest F1-score (0.8235) but required GPU resources.  
- **TF-IDF + SVM** provided the best CPU performance (0.7971 F1) and is suitable for resource-constrained environments.  

### 2. Improvement Strategies
- **Class Imbalance**: Use weighted loss functions or oversampling techniques like SMOTE.  
- **Feature Engineering**: Incorporate additional text features (e.g., emojis, sentiment scores).  
- **Ensemble Methods**: Combine predictions from top-performing models (e.g., TF-IDF SVM + DistilBERT).  

### 3. Challenges Faced
- **Preprocessing Complexity**: Balancing thorough cleaning with retaining useful information (e.g., hashtags).  
- **Hardware Limitations**: Transformer fine-tuning required GPU for training.  

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
