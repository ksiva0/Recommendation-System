import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from warnings import filterwarnings

filterwarnings('ignore')

@st.cache_data
def load_data():
    df = pd.read_excel(r"student-scores.csv")
    return df

# Main Streamlit app
def main():
    st.title("Subject Recommendation System")
    st.image('https://img.freepik.com/free-vector/happy-students-jump-joy_107791-12097.jpg?ga=GA1.1.1985650128.1724917128&semt=ais_hybrid')

    # Load the dataset
    df = load_data()

    # Data processing
    subject_columns = ['math_score', 'history_score', 'physics_score', 'chemistry_score', 'biology_score', 'english_score', 'geography_score']
    df['total_score'] = df[subject_columns].sum(axis=1)
    df['average_score'] = df['total_score'] / len(subject_columns)
    
    label_encoder = LabelEncoder()
    X = df[subject_columns].values
    y = label_encoder.fit_transform(df['career_aspiration'])
    y = to_categorical(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Model architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    
    # Compile and train the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_scaled, y, epochs=600, batch_size=20, validation_split=0.2, verbose=0)
    
    # Show training results
    st.subheader("Training Accuracy and Loss")
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    
    st.pyplot(fig)

    # Make a prediction
    st.subheader("Predict Career Aspiration")
    student_id = st.number_input("Enter Student ID", min_value=int(df['id'].min()), max_value=int(df['id'].max()))
    
    if st.button("Predict"):
        student_scores = df[df['id'] == student_id][subject_columns].values
        if student_scores.size > 0:
            student_scores_scaled = scaler.transform(student_scores)
            predicted_career = model.predict(student_scores_scaled)
            predicted_career_label = label_encoder.inverse_transform([predicted_career.argmax()])
            st.success(f"Recommended career aspiration for student {student_id}: {predicted_career_label[0]}")
        else:
            st.error("Invalid Student ID")

if __name__ == "__main__":
    main()
