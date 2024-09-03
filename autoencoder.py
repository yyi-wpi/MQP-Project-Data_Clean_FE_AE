import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Separate features and target
    X = df.drop(['Label', 'Attack'], axis=1)
    y = df['Attack']  # Using 'Attack' as the target variable

    # Encode categorical variables
    le = LabelEncoder()
    X['IPV4_SRC_ADDR'] = le.fit_transform(X['IPV4_SRC_ADDR'])
    X['IPV4_DST_ADDR'] = le.fit_transform(X['IPV4_DST_ADDR'])

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode target variable
    y_encoded = le.fit_transform(y)

    print(f"Shape of X: {X_scaled.shape}")
    print(f"Shape of y: {y_encoded.shape}")
    print(f"Number of unique classes in y: {len(np.unique(y_encoded))}")

    return X_scaled, y_encoded, X.columns


def create_autoencoder(input_dim, encoding_dim=32):
    input_layer = tf.keras.layers.Input(shape=(input_dim,))

    # Encoder
    encoded = tf.keras.layers.Dense(64, activation='relu')(input_layer)
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)

    # Decoder
    decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
    decoded = tf.keras.layers.Dense(input_dim, activation='linear')(decoded)

    # Autoencoder
    autoencoder = tf.keras.models.Model(input_layer, decoded)

    # Separate encoder
    encoder = tf.keras.models.Model(input_layer, encoded)

    return autoencoder, encoder


def train_autoencoder(autoencoder, X_train, X_val, epochs=50, batch_size=32):
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    return history


def create_classifier(input_dim, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def train_classifier(classifier, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    history = classifier.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    return history


def plot_training_history(history, output_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    output_dir = 'autoencoder_results'
    os.makedirs(output_dir, exist_ok=True)

    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data('NF-UNSW-NB15-v2_cleaned.csv')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train autoencoder
    autoencoder, encoder = create_autoencoder(X.shape[1])
    history = train_autoencoder(autoencoder, X_train, X_test)

    # Plot autoencoder training history
    plot_training_history(history, os.path.join(output_dir, 'autoencoder_training_history.png'))

    # Generate encoded representations
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    # Train classifier on encoded representations
    num_classes = len(np.unique(y))
    classifier = create_classifier(X_train_encoded.shape[1], num_classes)
    classifier_history = train_classifier(classifier, X_train_encoded, y_train, X_test_encoded, y_test)

    # Plot classifier training history
    plot_training_history(classifier_history, os.path.join(output_dir, 'classifier_training_history.png'))

    # Evaluate classifier
    y_pred = np.argmax(classifier.predict(X_test_encoded), axis=1)
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, os.path.join(output_dir, 'confusion_matrix.png'))

    print(f"All results and models have been saved to {output_dir}")


if __name__ == "__main__":
    main()