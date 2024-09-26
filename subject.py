import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from warnings import filterwarnings
filterwarnings('ignore')

# Load data with caching to avoid reloading on every interaction
@st.cache_data
def load_data():
    df = pd.read_excel(r"students_assessment.xlsx", index_col=[0, 1, 2], header=[0, 1])
    return df

# Streamlit app code
def main():
    st.title("Subject Recommendation System")

    st.image('https://img.freepik.com/free-vector/happy-students-jump-joy_107791-12097.jpg?ga=GA1.1.1985650128.1724917128&semt=ais_hybrid')
    # Load the dataset
    df = load_data()
    df = df.fillna(0)
    df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
    df.columns = [''.join(filter(None, col)).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    df.rename(columns={'level_0': "STUDENT NAME", "level_1": "INDEX NUMBER", "level_2": "TRACK"}, inplace=True)
    df.rename(columns={'ENGLISH_AVE':'ENGLISH', 'KISWAHILI_AVE':'KISWAHILI', 'MATH_AVE':'MATH', 'BIOLOGY_AVE':'BIOLOGY', 
                   'PHYSICS_AVE':'PHYSICS', 'CHEMISTRY_AVE':'CHEMISTRY', 'HIST_AVE':'HISTORY', 'GEO_AVE':'GEOLOGY',
                  'CRE_AVE':'CRE', 'C/st_AVE':'CST', 'BST_AVE':'BST'}, inplace=True)

    subjects = {
        "ENGLISH": ["ENGLISH_PP1_60", "ENGLISH_PP2_80", "ENGLISH_PP3_60", "ENGLISH", "ENGLISH_PTS"],
        "KISWAHILI": ["KISWAHILI_PP1_40", "KISWAHILI_PP2_80", "KISWAHILI_PP3_80", "KISWAHILI", "KISWAHILI_PTS"],
        "MATH": ["MATH_PP1_100", "MATH_PP2_100", "MATH", "MATH_PTS"],
        "BIOLOGY": ["BIOLOGY_PP1_80", "BIOLOGY_PP2_80", "BIOLOGY_PP3_40", "BIOLOGY", "BIOLOGY_PTS"],
        "PHYSICS": ["PHYSICS_PP1_80", "PHYSICS_PP2_80", "PHYSICS_PP3_40", "PHYSICS", "PHYSICS_PTS"],
        "CHEMISTRY": ["CHEMISTRY_PP1_80", "CHEMISTRY_PP2_80", "CHEMISTRY_PP3_40", "CHEMISTRY", "CHEMISTRY_PTS"],
        "HIST": ["HIST_PP1_100", "HIST_PP2_100", "HISTORY", "HIST_PTS"],
        "GEO": ["GEO_PP1_100", "GEO_PP2_100", "GEOLOGY", "GEO_PTS"],
        "CRE": ["CRE_PP1_100", "CRE_PP2_100", "CRE", "CRE_PTS"],
        "C/st": ["C/st_PP1_100", "C/st_PP2_100", "CST", "C/st_PTS"],
        "BST": ["BST_PP1_100", "BST_PP2_100", "BST", "BST_PTS"],
    }

    for subject, columns in subjects.items():
        for col in columns:
            if col in df.columns:
                df[col] = df[col].astype(int)
            else:
                print(f"Column {col} does not exist in the DataFrame for {subject}.")
    
    #st.write("Here is the student assessment data:")
    #st.dataframe(df)

    # Encoding and preparing the data for the model
    df['GRADE_POINTS'] = df['GRADE_POINTS'].astype(str)
    label_encoder = LabelEncoder()
    
    # Label encode the GRADE_POINTS column
    y = label_encoder.fit_transform(df['GRADE_POINTS'])
    y = to_categorical(y)
    
    # Prepare the features (subject scores)
    subject_columns = ['ENGLISH_AVE', 'KISWAHILI_AVE', 'MATH_AVE', 'BIOLOGY_AVE', 'PHYSICS_AVE', 
                   'CHEMISTRY_AVE', 'HIST_AVE', 'GEO_AVE', 'CRE_AVE', 'C/st_AVE', 'BST_AVE']
    X = df[subject_columns]
    
    # Define and train the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=150, batch_size=32, verbose=0)

    # Student ID selection and recommendation system
    student_id = st.selectbox("Select Student ID", df['INDEX NUMBER'].unique())
    threshold = st.slider("Select the threshold for subject improvement", 0, 100, 50)

    if st.button("Get Recommendation"):
        recommend_subjects(student_id, threshold, df, subject_columns, model, label_encoder)

# Recommendation function
def recommend_subjects(student_id, threshold, df, subject_columns, model, label_encoder):
    student_scores = df[df['INDEX NUMBER'] == student_id][subject_columns].values
    student_scores = student_scores.reshape(1, -1)

    grade = model.predict(student_scores)
    grade_label = label_encoder.inverse_transform([grade.argmax()])
    
    st.write(f"**Predicted Grade for Student {student_id}:** {grade_label[0]}")

    # Get the actual scores of the student
    student_scores = df[df['INDEX NUMBER'] == student_id][subject_columns].values.flatten()

    # Subjects with scores below the threshold
    low_score_subjects = [subject_columns[i] for i, score in enumerate(student_scores) if score < threshold]
    
    if low_score_subjects:
        st.write(f"**Subjects to improve for Student {student_id}:** {', '.join(low_score_subjects)}")
    else:
        st.write(f"**Student {student_id} is performing well in all subjects!**")

if __name__ == "__main__":
    main()
