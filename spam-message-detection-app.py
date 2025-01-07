
import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load model and vectorizer
tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Streamlit app customization
st.set_page_config(page_title="SMS Spam Detector", page_icon="📱", layout="wide")

# Title and Header Section
st.markdown(
    """
    <div style="background-color: #4CAF50; padding: 10px; border-radius: 10px;">
        <h1 style="color: white; text-align: center;">📩 SMS Spam Detection System</h1>
        <p style="color: white; text-align: center;">Built with Natural Language Processing (NLP) and Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Input Section
st.write("### Enter an SMS to predict if it's Spam or Not")
input_sms = st.text_area(
    label="Type your SMS message below:",
    height=150,
    placeholder="e.g., Congratulations! You've won a $1,000 gift card. Click here to claim your prize.",
)

# Predict button
if st.button("🚀 Predict"):
    if input_sms.strip():
        # Preprocess the SMS
        transformed_sms = transform_text(input_sms)

        # Vectorize the input
        vector_input = tk.transform([transformed_sms])

        # Make the prediction
        result = model.predict(vector_input)[0]

        # Display results
        if result == 1:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h2 style="color: red;">⚠️ Spam</h2>
                    <p style="color: red;">This message is likely spam. Be cautious!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="text-align: center;">
                    <h2 style="color: green;">✅ Not Spam</h2>
                    <p style="color: green;">This message seems legitimate.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.error("⚠️ Please enter a valid SMS message to predict.")

# Sidebar Section
with st.sidebar:
    st.image("https://via.placeholder.com/300x100.png?text=Edunet+Foundation", caption="Edunet Foundation", use_column_width=True)
    st.markdown(
        """
        ### About this App
        - **Project:** SMS Spam Detection
        - **Technology:** NLP + Machine Learning
        - **Framework:** Streamlit
        - **Developed by:** Rushikesh Lokhande
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Follow Us")
    st.write("[GitHub](https://github.com/) | [LinkedIn](https://linkedin.com/) | [Twitter](https://twitter.com/)")

# Footer Section
st.markdown(
    """
    <hr>
    <p style="text-align: center;">© 2024 Edunet Foundation. All rights reserved.</p>
    """,
    unsafe_allow_html=True,
)
