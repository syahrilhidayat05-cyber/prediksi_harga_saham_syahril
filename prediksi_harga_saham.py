import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import plotly.graph_objects as go
import time

# ==============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="Analisis & Prediksi Saham",
    page_icon="📈",
    layout="wide"
)

# ==============================================================================
# FUNGSI-FUNGSI BANTUAN DAN KELAS CALLBACK
# ==============================================================================

@st.cache_data
def load_data(ticker, start_date, end_date):
    """Mengunduh data saham dari Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if data.empty:
        return None
    data.reset_index(inplace=True)
    return data

def create_sequences(data, time_steps=60):
    """Membentuk data menjadi sekuens/urutan untuk input model LSTM/GRU."""
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

def build_model(model_type, input_shape, units, dropout, learning_rate, optimizer_name):
    """Membangun arsitektur model deep learning (LSTM atau GRU)."""
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    else: # GRU
        model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    if model_type == 'LSTM':
        model.add(LSTM(units=units, return_sequences=False))
    else:
        model.add(GRU(units=units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))

    if optimizer_name == 'Adam': optimizer = Adam(learning_rate=learning_rate)
    else: optimizer = optimizer_name
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

class StreamlitCallback(Callback):
    """Callback Keras untuk mengupdate progress bar di Streamlit."""
    def __init__(self, progress_bar, progress_text_placeholder, total_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.progress_text_placeholder = progress_text_placeholder
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch += 1
        progress = self.current_epoch / self.total_epochs
        self.progress_bar.progress(progress)
        self.progress_text_placeholder.text(f"Pelatihan sedang berjalan... Epoch {self.current_epoch}/{self.total_epochs}")

# Inisialisasi session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.results = None
    st.session_state.ticker = ""

# ==============================================================================
# SIDEBAR - PANEL KONTROL DENGAN HELP TEXT
# ==============================================================================
st.sidebar.title("⚙️ Panel Kontrol")
st.sidebar.markdown("---")
st.sidebar.header("1. Input Data")
ticker_input = st.sidebar.text_input("Kode Saham (e.g., BBCA.JK)", "BBCA.JK")
start_date = st.sidebar.date_input("Tanggal Mulai", pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("Tanggal Selesai", pd.to_datetime('today'))
st.sidebar.markdown("---")
st.sidebar.header("2. Pengaturan Pelatihan Model")
model_type = st.sidebar.selectbox("Pilih Arsitektur Model", ('LSTM', 'GRU'))
time_steps = st.sidebar.slider("Jendela Waktu (Time Steps)", 10, 100, 60, help="Jumlah hari data historis yang digunakan model untuk memprediksi harga pada hari berikutnya. Nilai 60 berarti model melihat data 60 hari terakhir untuk membuat 1 prediksi.")
split_ratio = st.sidebar.slider("Rasio Data Training (%)", 70, 95, 80, help="Persentase dari total data historis yang digunakan untuk 'melatih' model. Sisa datanya digunakan untuk 'menguji' seberapa baik performa model.")

with st.sidebar.expander("Hiperparameter Lanjutan"):
    units = st.slider("Jumlah Unit di Layer", 30, 200, 50, help="Menentukan kompleksitas dari layer model (LSTM/GRU). Jumlah unit yang lebih besar dapat menangkap pola yang lebih rumit, namun juga meningkatkan risiko overfitting dan waktu pelatihan.")
    epochs = st.slider("Jumlah Epoch", 10, 200, 50, 10, help="Berapa kali model akan melihat keseluruhan data training selama proses pelatihan. Terlalu sedikit bisa membuat model kurang belajar (underfitting), terlalu banyak bisa menyebabkan overfitting.")
    batch_size = st.slider("Ukuran Batch", 8, 128, 32, help="Jumlah data sampel yang diproses oleh model sebelum bobotnya diperbarui. Ukuran batch yang lebih kecil membutuhkan lebih banyak waktu namun bisa lebih stabil.")
    learning_rate = st.number_input("Learning Rate", 0.0001, 0.1, 0.001, format="%.4f", help="Mengontrol seberapa besar penyesuaian yang dilakukan pada model di setiap pembaruan. Nilai yang terlalu tinggi bisa tidak stabil, nilai yang terlalu rendah bisa sangat lambat.")
    dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05, help="Teknik untuk mencegah overfitting. Nilai 0.2 berarti 20% dari unit di layer akan diabaikan secara acak selama pelatihan, memaksa model belajar pola yang lebih kuat.")
    optimizer_name = st.selectbox("Optimizer", ('Adam', 'RMSprop', 'SGD'), help="Algoritma yang digunakan untuk mengubah bobot model guna mengurangi error (loss). 'Adam' adalah pilihan umum yang bekerja dengan baik di banyak kasus.")
st.sidebar.markdown("---")

st.sidebar.header("3. Pengaturan Prediksi")
prediction_days = st.sidebar.slider("Hari Prediksi ke Depan", 1, 60, 1)

train_button = st.sidebar.button("🚀 Latih Model & Prediksi!", use_container_width=True)

# ==============================================================================
# HALAMAN UTAMA - TAMPILAN
# ==============================================================================
st.title(f"📈 Analisis & Prediksi Harga Saham: {ticker_input.upper()}")
st.markdown("Gunakan panel kontrol di sebelah kiri untuk mengatur data dan parameter model.")
placeholder = st.empty()

if train_button:
    st.session_state.ticker = ticker_input
    with st.spinner(f"Mengunduh data untuk {st.session_state.ticker}..."):
        stock_data = load_data(st.session_state.ticker, start_date, end_date)
    if stock_data is None:
        st.error("Gagal memuat data. Periksa kembali kode saham atau rentang tanggal.")
    else:
        # Proses pra-pemrosesan dan pelatihan model (tidak diubah)
        with st.spinner("Melakukan pra-pemrosesan data..."):
            close_prices = stock_data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            training_data_len = int(np.ceil(len(scaled_data) * (split_ratio / 100)))
            train_data = scaled_data[0:int(training_data_len), :]
            X_train, y_train = create_sequences(train_data, time_steps)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            test_data_for_sequences = scaled_data[training_data_len - time_steps:, :]
            X_test, y_test = create_sequences(test_data_for_sequences, time_steps)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        st.info(f"Memulai pelatihan model {model_type}. Ini mungkin memakan waktu.")
        model = build_model(model_type, (X_train.shape[1], 1), units, dropout, learning_rate, optimizer_name)

        progress_text_placeholder = st.empty()
        progress_bar = st.progress(0)
        st_callback = StreamlitCallback(progress_bar, progress_text_placeholder, epochs)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0, callbacks=[st_callback])

        progress_text_placeholder.empty()
        progress_bar.empty()

        with st.spinner("Menghitung prediksi dan metrik..."):
            predictions_test = model.predict(X_test, verbose=0)
            predictions_test = scaler.inverse_transform(predictions_test)
            y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

            predictions_train = model.predict(X_train, verbose=0)
            predictions_train = scaler.inverse_transform(predictions_train)
            y_train_unscaled = scaler.inverse_transform(y_train.reshape(-1, 1))

            mape_test = np.mean(np.abs((y_test_unscaled - predictions_test) / (y_test_unscaled + 1e-10))) * 100
            accuracy_test = max(0, 100 - mape_test)
            mape_train = np.mean(np.abs((y_train_unscaled - predictions_train) / (y_train_unscaled + 1e-10))) * 100
            accuracy_train = max(0, 100 - mape_train)

            future_predictions = []
            last_sequence_full_data = scaled_data[-time_steps:]
            current_sequence = last_sequence_full_data.reshape((1, time_steps, 1))
            for _ in range(prediction_days):
                next_pred_scaled = model.predict(current_sequence, verbose=0)
                future_predictions.append(next_pred_scaled[0,0])
                new_sequence = np.append(current_sequence[0, 1:, 0], next_pred_scaled)
                current_sequence = new_sequence.reshape((1, time_steps, 1))
            future_predictions_unscaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        st.session_state.model_trained = True
        st.session_state.results = {
            "stock_data": stock_data, "training_len": training_data_len,
            "predictions_test": predictions_test, "mape_test": mape_test, "accuracy_test": accuracy_test,
            "mape_train": mape_train, "accuracy_train": accuracy_train,
            "history": history.history, "future_predictions": future_predictions_unscaled,
            "time_steps": time_steps, "prediction_days": prediction_days
        }
        st.success("🎉 Pelatihan dan prediksi selesai!")
        time.sleep(1); placeholder.empty()

if st.session_state.model_trained:
    results = st.session_state.results
    stock_data = results["stock_data"]
    tab1, tab2, tab3 = st.tabs(["📊 Dasbor Hasil", "📉 Kinerja Model", "📚 Data Mentah"])

    with tab1:
        st.header(f"Hasil Prediksi untuk {st.session_state.ticker.upper()}")

        st.markdown("#### Ringkasan Kinerja & Harga")
        latest_price = float(stock_data['Close'].iloc[-1])
        model_mape_test = float(results['mape_test'])
        model_accuracy_test = float(results['accuracy_test'])
        model_accuracy_train = float(results['accuracy_train'])

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Harga Terakhir", f"Rp {latest_price:,.2f}")
        col2.metric("Akurasi Model (Uji)", f"{model_accuracy_test:.2f}%", help="Akurasi prediksi pada data yang belum pernah dilihat model (100% - MAPE Uji).")
        col3.metric("MAPE Model (Uji)", f"{model_mape_test:.2f}%", help="Rata-rata persentase error pada data uji. Semakin kecil semakin baik.")
        col4.metric("Akurasi Model (Latih)", f"{model_accuracy_train:.2f}%", help="Akurasi prediksi pada data yang sudah dilihat model saat pelatihan.")
        st.markdown("---")

        st.markdown(f"#### Prediksi untuk {results['prediction_days']} Hari ke Depan")
        future_preds_values = results['future_predictions'].flatten()
        
        # --- BLOK LOGIKA BARU UNTUK METRIK DENGAN IKON ---
        # Logika untuk kesimpulan tren jangka panjang (berdasarkan hari terakhir prediksi)
        if future_preds_values[-1] > latest_price:
            trend = "Cenderung NAIK"
        else:
            trend = "Cenderung TURUN"
        
        # Logika untuk ikon dan teks perubahan harga besok
        pred_1_day = future_preds_values[0]
        delta_1_day = pred_1_day - latest_price

        if pred_1_day >= latest_price:
            icon = "📈" # Ikon grafik naik
            change_text = f"Naik Rp {delta_1_day:,.2f} ({delta_1_day/latest_price:.2%})"
        else:
            icon = "📉" # Ikon grafik turun
            change_text = f"Turun Rp {abs(delta_1_day):,.2f} ({abs(delta_1_day)/latest_price:.2%})"

        # Tampilkan metrik dengan ikon, tanpa delta bawaan
        st.metric(
            label=f"Prediksi Harga Besok {icon}",
            value=f"Rp {pred_1_day:,.2f}",
            help="Menunjukkan prediksi harga untuk hari perdagangan berikutnya."
        )
        # Tampilkan perubahan sebagai teks keterangan (caption) di bawah metrik
        st.caption(change_text)

        st.info(f"**Kesimpulan Tren:** Berdasarkan prediksi untuk {results['prediction_days']} hari ke depan, harga saham **{st.session_state.ticker.upper()}** menunjukkan tren **{trend}**.")
        st.markdown("---")
        
        st.markdown("#### Rincian Tabel Prediksi")
        last_date_in_data = stock_data['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date_in_data + pd.Timedelta(days=1), periods=results['prediction_days'], freq='B')
        
        future_df = pd.DataFrame({
            'Tanggal': future_dates,
            'Prediksi Harga (Rp)': future_preds_values
        })
        
        future_df['Perubahan dari Harga Terakhir (Rp)'] = future_df['Prediksi Harga (Rp)'] - latest_price
        future_df['Perubahan (%)'] = (future_df['Perubahan dari Harga Terakhir (Rp)'] / latest_price) * 100

        future_df['Prediksi Harga (Rp)'] = future_df['Prediksi Harga (Rp)'].apply(lambda x: f"{x:,.2f}")
        future_df['Perubahan dari Harga Terakhir (Rp)'] = future_df['Perubahan dari Harga Terakhir (Rp)'].apply(lambda x: f"{x:,.2f}")
        future_df['Perubahan (%)'] = future_df['Perubahan (%)'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(future_df.set_index('Tanggal'), use_container_width=True)
        st.markdown("---")

        with st.expander("Lihat Ringkasan Statistik Data Historis per Tahun"):
            st.markdown("Statistik ini dihitung dari data historis yang dikelompokkan per tahun untuk melihat tren dan perubahan dari waktu ke waktu.")
            
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            unique_years = sorted(stock_data['Date'].dt.year.unique(), reverse=True)
            
            for year in unique_years:
                st.subheader(f"Statistik untuk Tahun {year}")
                yearly_data = stock_data[stock_data['Date'].dt.year == year]
                
                stats_df = yearly_data[['Close']].describe().T.rename(columns={
                    'count': 'Jumlah Hari Perdagangan',
                    'mean': 'Rata-rata Harga (Rp)',
                    'std': 'Standar Deviasi (Volatilitas)',
                    'min': 'Harga Terendah (Rp)',
                    '25%': 'Kuartil 1 (25%)',
                    '50%': 'Median (50%)',
                    '75%': 'Kuartil 3 (75%)',
                    'max': 'Harga Tertinggi (Rp)'
                })
                
                st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)
                if year != unique_years[-1]:
                    st.markdown("---")

    with tab2:
        st.header("Analisis Kinerja Pelatihan Model")
        st.subheader("Grafik Loss Pelatihan vs. Validasi")
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=results['history']['loss'], name='Training Loss', line=dict(color='blue')))
        loss_fig.add_trace(go.Scatter(y=results['history']['val_loss'], name='Validation Loss', line=dict(color='orange')))
        loss_fig.update_layout(title='Loss Selama Pelatihan', xaxis_title='Epoch', yaxis_title='Loss')
        st.plotly_chart(loss_fig, use_container_width=True)
        st.info("""**Bagaimana cara membaca grafik ini?**\n- **Training Loss** (biru) harus menurun. Ini menunjukkan model belajar.\n- **Validation Loss** (oranye) juga harus menurun. Jika validation loss mulai naik sementara training loss terus turun, ini adalah tanda **overfitting**.\n- Model yang baik adalah ketika kedua kurva sama-sama menurun dan bertemu di titik terendah.""")

    with tab3:
        st.header("Data Historis Mentah")
        st.dataframe(stock_data)
else:
    placeholder.info("Atur parameter pada sidebar di sebelah kiri dan klik **'Latih Model & Prediksi!'** untuk memulai analisis.")

st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini dibuat untuk tujuan edukasi. **Ini bukan merupakan saran finansial.**")
