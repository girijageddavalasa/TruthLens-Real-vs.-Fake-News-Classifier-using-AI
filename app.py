import streamlit as st
import pandas as pd
import joblib, json, time, string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_lottie import st_lottie

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

# Lottie loader
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Load Lottie animations
read_anim = load_lottie("assets/People_reading.json")
check_anim = load_lottie("assets/Check_Mark.json")
confetti_anim = load_lottie("assets/Confetti_Effects.json")
social_anim = load_lottie("assets/Social_Icons.json")

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    tokens = text.split()
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Load CSVs from Google Drive
@st.cache_data
def load_csv_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    return pd.read_csv(url)

# Combine True + Fake datasets
@st.cache_resource
def train_model():
    # Google Drive File IDs
    TRUE_ID = "17x0QJfS4uFBgqX1i8VjTzUA-8eTyU9gg"
    FAKE_ID = "11bHSUmuJ05BtHFNR6H8inYFeYkg_R29C"

    df_true = load_csv_from_drive(TRUE_ID)
    df_fake = load_csv_from_drive(FAKE_ID)

    df_true["label"] = 1  # Real
    df_fake["label"] = 0  # Fake

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['content'] = df['content'].apply(preprocess)

    X = df['content']
    y = df['label']

    tfidf = TfidfVectorizer(max_features=5000)
    X_vec = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")
    joblib.dump(tfidf, "vectorizer.pkl")

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, tfidf, acc

# Load trained model and vectorizer
model, vectorizer, accuracy = train_model()

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("🧠 Real vs. Fake News Detector")
st_lottie(read_anim, height=250)

user_input = st.text_area("📄 Paste your news headline or article below:", height=200)

if st.button("🔎 Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a news article.")
    else:
        with st.spinner("🤖 Analyzing..."):
            time.sleep(1.2)
            cleaned = preprocess(user_input)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            conf = model.predict_proba(vec)[0][pred] * 100

        st.subheader("🔍 Result")
        if pred == 0:
            st.error(f"🛑 **Fake News Detected** with `{conf:.2f}%` confidence.")
        else:
            st.success(f"✅ **Real News Detected** with `{conf:.2f}%` confidence.")
            st_lottie(check_anim, height=150)

        if conf > 90:
            st_lottie(confetti_anim, height=200)

        meter = "😐" if conf < 60 else "🙂" if conf < 85 else "🔥"
        st.markdown(f"**Confidence Meter:** {meter}  `{conf:.2f}%`")

        with st.expander("🔍 What words influenced this prediction?"):
            words = cleaned.split()
            vocab = vectorizer.vocabulary_
            top = [(w, vocab[w]) for w in words if w in vocab]
            top = sorted(top, key=lambda x: x[1], reverse=True)[:5]
            for w, idx in top:
                st.markdown(f"- **{w}** (TF-IDF index: `{idx}`)")

        with st.expander("🧪 Try These Funny Fake Headlines"):
            for example in [
                "Aliens take over United Nations headquarters",
                "Scientists confirm coffee cures all diseases",
                "Government plans to give every citizen a private jet",
            ]:
                st.code(example)

# Sidebar
st.sidebar.header("⚙️ Settings")
theme = st.sidebar.radio("Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""<style>body{background:#111;color:#eee;}</style>""", unsafe_allow_html=True)

st.sidebar.write(f"📊 Accuracy: **{accuracy*100:.2f}%**")
st.sidebar.markdown("---")
st.sidebar.subheader("📢 Follow Us")
st_lottie(social_anim, height=180)
