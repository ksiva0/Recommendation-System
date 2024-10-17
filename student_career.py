import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from warnings import filterwarnings
filterwarnings('ignore')

@st.cache_data
def load_data():
  df = pd.read_excel(r"student-scores.csv")
  return df

def main():
  st.title("Subject Recommendation System")

  st.image('https://img.freepik.com/free-vector/happy-students-jump-joy_107791-12097.jpg?ga=GA1.1.1985650128.1724917128&semt=ais_hybrid')
  # Load the dataset
  df = load_data()
  df['total_score'] = df['math_score'] + df['history_score'] + df['physics_score'] +df['chemistry_score'] + df['biology_score'] + df['english_score'] + df['geography_score']
  df['average_score'] = df['total_score'] / 7

  X = df[subject_columns].values
  y = label_encoder.fit_transform(df['career_aspiration'])
  y = to_categorical(y)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # Build the improved model
  model = Sequential([
      Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
      Dropout(0.2),  # Dropout layer to prevent overfitting
      Dense(128, activation='relu'),
      Dropout(0.2),
      Dense(64, activation='relu'),
      Dense(32, activation='relu'),
      Dense(y.shape[1], activation='softmax')  # Output layer with softmax for multiclass classification
  ])

  # Compile the model with a lower learning rate
  optimizer = Adam(learning_rate=0.001)
  model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
  
  # Train the model with more epochs and a larger batch size
  history = model.fit(X_scaled, y, epochs=600, batch_size=20, validation_split=0.2)

  # Make predictions for a student after scaling their data
  student_id = 10
  student_scores = df[df['id'] == student_id][subject_columns].values
  student_scores_scaled = scaler.transform(student_scores)
  
  # Predict the career aspiration
  predicted_career = model.predict(student_scores_scaled)
  predicted_career_label = label_encoder.inverse_transform([predicted_career.argmax()])
  
  print(f"Recommended career aspiration for student {student_id}: {predicted_career_label[0]}")
