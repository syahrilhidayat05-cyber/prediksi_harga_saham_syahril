import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==============================
# Fungsi ambil data sesuai horizon
# ==============================
def load_data(ticker, horizon):
    if horizon == "1 Hari":
        return yf.download(ticker, period="1y", interval="1d")
    elif horizon == "1 Minggu":
        return yf.download(ticker, period="3y", interval="1wk")
    elif horizon == "1 Bulan":
        return yf.download(ticker, period="7y", interval="1mo")

# ==============================
# Fungsi siapkan data untuk LSTM (window fleksibel)
# ==============================
def prepare_data(data, window=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.reshape(-1, 1))

    # kalau data lebih sedikit dari window, turunkan window
    if len(scaled) < window:
        window = max(2, len(scaled) - 1)  # minimal butuh 2 titik data

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler, window

# ==============================
# Fungsi buat model LSTM
# ==============================
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# ==============================
# Fungsi multi-step forecast
# ==============================
def forecast_multi_step(model, last_window, scaler, steps=10):
    predictions = []
    current_input = last_window.copy()

    for _ in range(steps):
        X_test = np.reshape(current_input, (1, current_input.shape[0], 1))
        pred_scaled = model.predict(X_test, verbose=0)
        pred = scaler.inverse_transform(pred_scaled)[0][0]
        predictions.append(pred)

        # update input window
        new_scaled = scaler.transform(np.array([[pred]]))
        current_input = np.append(current_input[1:], new_scaled).reshape(-1,1)

    return predictions

# ==============================
# Streamlit UI
# ==============================
st.title("📈 Prediksi Saham Multi-Step dengan LSTM")

ticker = st.text_input("Masukkan kode saham (contoh: BBRI.JK, TLKM.JK, AAPL):", "BBRI.JK")
horizon = st.radio("Pilih Horizon Prediksi:", ["1 Hari", "1 Minggu", "1 Bulan"])
steps = st.slider("Jumlah langkah prediksi ke depan:", 1, 20, 10)

if st.button("🔮 Prediksi"):
    with st.spinner("Sedang mengambil data dan training model..."):
        data = load_data(ticker, horizon)
        if data.empty:
            st.error("❌ Data saham tidak ditemukan!")
        else:
            close_data = data["Close"].values

            # siapkan data dengan window adaptif
            X, y, scaler, window_used = prepare_data(close_data)

            if len(X) == 0:
                st.error("❌ Data saham terlalu sedikit untuk dibuat model.")
            else:
                model = build_model((X.shape[1], 1))
                model.fit(X, y, epochs=10, batch_size=16, verbose=0)

                # Ambil window terakhir sesuai ukuran window yang dipakai
                last_window = scaler.transform(close_data[-window_used:].reshape(-1, 1))

                # Prediksi multi-step sesuai pilihan user
                preds = forecast_multi_step(model, last_window, scaler, steps=steps)

                # ==============================
                # Tampilkan hasil angka
                # ==============================
                st.success(f"💹 Prediksi harga {ticker} ({horizon.lower()}) {steps} langkah ke depan:")
                for i, p in enumerate(preds, 1):
                    st.write(f"Langkah {i}: Rp {p:,.2f}")

                # ==============================
                # Analisis sederhana untuk orang awam
                # ==============================
                start_price = preds[0]
                end_price = preds[-1]
                max_price = max(preds)
                max_step = preds.index(max_price) + 1

                if max_step == 1 and end_price < start_price:
                    advice = "📉 Harga diperkirakan langsung turun setelah ini. Sebaiknya hati-hati jika ingin beli."
                elif max_step < len(preds) and end_price < max_price:
                    advice = f"📈 Harga diperkirakan naik hingga langkah {max_step} (Rp {max_price:,.2f}), lalu mulai turun. Pertimbangkan jual sebelum turun."
                elif end_price > start_price:
                    advice = "📈 Harga diperkirakan terus naik pada periode ini. Cocok untuk hold jangka pendek."
                else:
                    advice = "🔄 Harga relatif stabil, tidak ada tren jelas."

                st.info(advice)

                # ==============================
                # Plot hasil prediksi + tanda
                # ==============================
                future_index = pd.date_range(
                    start=data.index[-1], periods=steps+1, freq=data.index.inferred_freq
                )[1:]

                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data.index, close_data, label="Harga Aktual", color="blue")
                ax.plot(future_index, preds, "r--", marker="o", label=f"Prediksi {steps} langkah")

                # Tandai titik maksimum (potensi jual)
                ax.scatter(future_index[max_step-1], max_price, color="green", s=100, zorder=5, label="Potensi Jual (Harga Tertinggi)")

                # Tandai titik terakhir (akhir prediksi)
                ax.scatter(future_index[-1], end_price, color="red", s=100, zorder=5, label="Akhir Prediksi")

                ax.legend()
                ax.set_title(f"Prediksi {ticker} - {horizon}")
                st.pyplot(fig)
