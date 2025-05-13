# NER-LSTM

"""
# Arabic Named Entity Recognition

## Project Structure
- `Arabic_NER.ipynb`: Main project notebook
- `Ner.keras`: Saved model file
- `coversheet.pdf`: Project coversheet
- `presentation.pdf`: Project presentation

## Dataset
This project uses the Arabic NER dataset from SinaLab:
- Dataset source: https://github.com/SinaLab/ArabicNER
- The dataset is automatically downloaded when running the notebook

## Requirements
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Plotly
- Hugging Face Datasets
- livelossplot
- keras-contrib

## Model Architecture
- Bidirectional LSTM with Arabic FastText embeddings
- 300-dimensional word embeddings
- 64 LSTM units (128 total after bidirectional)
- TimeDistributed Dense layer for token classification

## Results
- Test accuracy: 95%
- #### High-Performing Entity Classes (AUC ≥ 0.98):
- Perfect classification (AUC = 1.00): I-ORDINAL, I-DATE, I-PERS, I-GPE, B-OCC, B-PERS, B-GPE, B-ORDINAL, B-DATE, I-MONEY, I-WEBSITE, I-CARDINAL
- Near-perfect classification (AUC ≥ 0.98): I-FAC (0.99), I-ORG (0.98), I-LOC (0.99), B-FAC (0.99), B-WEBSITE (0.98), B-ORG (0.99), B-CARDINAL (0.98), I-OCC (0.99)

#### Good-Performing Entity Classes (0.90 ≤ AUC < 0.98):
- I-EVENT (0.93), I-NORP (0.97), B-LANGUAGE (0.95), B-NORP (0.97), B-EVENT (0.95), B-LAW (0.95), B-LOC (0.97), B-TIME (0.92)

#### Lower-Performing Entity Classes (AUC < 0.90):
- B-MONEY (0.80), B-PRODUCT (0.79), I-TIME (0.85), I-PRODUCT (0.84)


## How to Run
1. Clone the repository
2. Open and run the notebook `Arabic_NER.ipynb`
3. The dataset will be automatically downloaded
"""
