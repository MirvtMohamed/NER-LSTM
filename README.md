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
- Test accuracy: [Add your final test accuracy]
- ROC-AUC scores for entity classes: [Add summary of ROC-AUC scores]

## How to Run
1. Clone the repository
2. Open and run the notebook `Arabic_NER.ipynb`
3. The dataset will be automatically downloaded
"""
