import streamlit as st
import numpy as np
import librosa
import joblib
import tensorflow as tf
import os
import soundfile as sf
import matplotlib.pyplot as plt

# --------------------------------- PARTE 1: EXTRAIR FEATURES --------------------------------- #

# Carregar o modelo e o scaler
MODEL_PATH = "C:\\Users\\caual\\OneDrive\\Documentos\\Área de Trabalho\\Resumos\\Trilha\\Mini Projeto 2\\miniprojeto2\\meu_modelo.keras"
SCALER_PATH = "C:\\Users\\caual\\OneDrive\\Documentos\\Área de Trabalho\\Resumos\\Trilha\\Mini Projeto 2\\data\\scaler.pkl" 

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de emoções
EMOTIONS = ["angry", "calm", "disgust", "fear",
            "happy", "neutral", "sad", "surprise"]


# Função para extrair features
def extract_features(audio_path):
    data, sr = librosa.load(audio_path, sr=16000, mono=True)
    features = []

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(data)
    features.extend(zcr.mean(axis=1))

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=data, sr=sr)
    features.extend(chroma.mean(axis=1))

    # MFCCs
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)
    features.extend(mfccs.mean(axis=1))

    # RMS
    rms = librosa.feature.rms(y=data)
    features.extend(rms.mean(axis=1))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=data, sr=sr)
    features.extend(mel.mean(axis=1))   

    # Garantir que tenha exatamente 162 features (ou truncar/zerar)
    target_length = 155
    if len(features) < target_length:
        features.extend([0] * (target_length - len(features)))
    elif len(features) > target_length:
        features = features[:target_length]

    return np.array(features).reshape(1, -1)


# --------------------------------- PARTE 2: STREAMLIT --------------------------------- #

# Configuração do app Streamlit (Título e descrição)
st.title("🎵Detector de Emoções em Áudio")
st.write("Envie um arquivo de áudio para análise!")

# Upload de arquivo de áudio (wav, mp3, ogg)
uploaded_file = st.file_uploader(
    "Escolha um arquivo de áudio...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Salvar temporariamente o áudio
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as f:
        f.write(uploaded_file.read())

    # Reproduzir o áudio enviado
    st.audio(temp_audio_path, format="audio/wav")

    # Extrair features
    features = extract_features(temp_audio_path)

    # Normalizar os dados com o scaler treinado
    normalized_features = scaler.transform(features)

    # Ajustar formato para o modelo
    input_data = normalized_features.reshape(1, -1)

    # Fazer a predição
    predictions = model.predict(normalized_features)
    predicted_emotion = EMOTIONS[np.argmax(predictions)]

    # Exibir o resultado
    st.write(f"🎭Emoção Detectada: {predicted_emotion}")

    # Exibir probabilidades (gráfico de barras)
    st.write("Probabilidades por Emoção:")
    fig, ax = plt.subplots()
    ax.bar(EMOTIONS, predictions[0], color="skyblue")
    ax.set_xlabel("Emoções")
    ax.set_ylabel("Probabilidade")
    ax.set_title("Distribuição de Probabilidades")
    st.pyplot(fig)

    # Remover o arquivo temporário
    os.remove(temp_audio_path)
