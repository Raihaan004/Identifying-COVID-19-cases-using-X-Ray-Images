import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, 
Dropout 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.callbacks import EarlyStopping 
 
# Constants 
CSV_FILE = 'covid_xray_metadata.csv' 
IMAGE_SIZE = (150, 150) 
BATCH_SIZE = 32 
EPOCHS = 20 
 
 
class COVIDXRayClassifier: 
    def __init__(self): 
        self.model = None 
        self.label_encoder = LabelEncoder() 
        self.class_names = ['covid', 'normal', 'pneumonia'] 
 
    def load_data(self, csv_file): 
        """Load data from CSV and prepare image paths and labels""" 
        df = pd.read_csv(csv_file) 
 
        # For demo purposes, we'll just use the paths and labels 
        # In a real scenario, you would load the actual images 
        print(f"Loaded {len(df)} records from {csv_file}") 
        print("Class distribution:") 
        print(df['label'].value_counts()) 
 
        return df['image_path'].values, df['label'].values 
 
    def preprocess_images(self, image_paths, labels): 
        """Load and preprocess images""" 
        X = [] 
        y = [] 
 
        # Convert labels to numerical values 
        y_encoded = self.label_encoder.fit_transform(labels) 
        y_categorical = to_categorical(y_encoded) 
 
        # Simulate loading images (in a real scenario, you would load actual 
images) 
        print("\nNote: In this demo, we're simulating image loading.") 
        print("In a real implementation, you would load images from the paths.") 
 
        # Generate random arrays to simulate image data 
        num_images = len(image_paths) 
        X_simulated = np.random.rand(num_images, *IMAGE_SIZE, 1)  # Grayscale 
        y_simulated = y_categorical 
 
        return X_simulated, y_simulated 
 
    def build_model(self, input_shape, num_classes): 
        """Build CNN model architecture""" 
        model = Sequential([ 
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape), 
            MaxPooling2D((2, 2)), 
 
            Conv2D(64, (3, 3), activation='relu'), 
            MaxPooling2D((2, 2)), 
 
            Conv2D(128, (3, 3), activation='relu'), 
            MaxPooling2D((2, 2)), 
 
            Flatten(), 
            Dense(128, activation='relu'), 
            Dropout(0.5), 
            Dense(num_classes, activation='softmax') 
        ]) 
 
        model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy']) 
 
        return model 
 
    def train(self, X_train, y_train, X_val, y_val): 
        """Train the model""" 
        input_shape = X_train.shape[1:] 
        num_classes = y_train.shape[1] 
 
        self.model = self.build_model(input_shape, num_classes) 
 
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, 
restore_best_weights=True) 
 
        history = self.model.fit( 
            X_train, y_train, 
            validation_data=(X_val, y_val), 
            batch_size=BATCH_SIZE, 
            epochs=EPOCHS, 
            callbacks=[early_stopping] 
        ) 
 
        return history 
 
    def evaluate(self, X_test, y_test): 
        """Evaluate model performance""" 
        if self.model is None: 
            raise ValueError("Model has not been trained yet.") 
 
        loss, accuracy = self.model.evaluate(X_test, y_test) 
        print(f"\nTest Accuracy: {accuracy:.2f}") 
        print(f"Test Loss: {loss:.2f}") 
 
        return loss, accuracy 
 
    def plot_history(self, history): 
        """Plot training history""" 
        plt.figure(figsize=(12, 4)) 
 
        plt.subplot(1, 2, 1) 
        plt.plot(history.history['accuracy'], label='Train Accuracy') 
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy') 
        plt.title('Accuracy over Epochs') 
        plt.legend() 
 
        plt.subplot(1, 2, 2) 
        plt.plot(history.history['loss'], label='Train Loss') 
        plt.plot(history.history['val_loss'], label='Validation Loss') 
        plt.title('Loss over Epochs') 
        plt.legend() 
 
        plt.tight_layout() 
        plt.show() 
 
 
def main(): 
    # Initialize classifier 
    classifier = COVIDXRayClassifier() 
 
    # Load data from CSV 
    image_paths, labels = classifier.load_data(CSV_FILE) 
 
    # Preprocess data (in a real scenario, this would load actual images) 
    X, y = classifier.preprocess_images(image_paths, labels) 
 
    # Split data into train, validation, and test sets 
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, 
stratify=y) 
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, 
stratify=y_temp) 
 
    print(f"\nData split:") 
    print(f"Training: {X_train.shape[0]} samples") 
    print(f"Validation: {X_val.shape[0]} samples") 
    print(f"Test: {X_test.shape[0]} samples") 
 
    # Train the model 
    print("\nTraining model...") 
    history = classifier.train(X_train, y_train, X_val, y_val) 
 
    # Evaluate the model 
    print("\nEvaluating model...") 
    classifier.evaluate(X_test, y_test) 
 
    # Plot training history 
    classifier.plot_history(history) 
 
    # Save the model (optional) 
    # classifier.model.save('covid_xray_classifier.h5') 
 
 
if __name__ == "__main__": 
    main() 