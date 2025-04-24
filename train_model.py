"""Script to train the model."""
from src.ml.train import train_model

if __name__ == '__main__':
    print("Starting model training...")
    model = train_model()
    print("Model training completed successfully!") 