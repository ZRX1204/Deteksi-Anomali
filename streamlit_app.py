import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Fungsi untuk membaca data dari Excel dan memproses anomali
def read_and_process_data():
    # Baca data dari file Excel
    data = pd.read_excel("/content/drive/My Drive/Colab Notebooks/Convert.xlsx")

    # Menggunakan LabelEncoder untuk mengubah fitur 'KELAMIN' menjadi angka
    label_encoder = LabelEncoder()
    data['KELAMIN'] = label_encoder.fit_transform(data['KELAMIN'])

    # Inisialisasi OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False)

    # Melakukan encoding one-hot untuk kolom 'KELAMIN'
    encoded = onehot_encoder.fit_transform(data['KELAMIN'].values.reshape(-1, 1))

    # Membuat dataframe baru dengan hasil encoding one-hot
    encoded_df = pd.DataFrame(encoded[:, :-1], columns=['L', 'P'])  # Mengambil semua kolom kecuali yang terakhir

    # Gabungkan dataframe baru dengan dataframe awal
    data = pd.concat([data, encoded_df], axis=1)

    # Pilih fitur yang akan digunakan untuk mendeteksi anomali
    fitur = ['USIA']

    # Preprocessing data (menghilangkan nilai NaN jika ada)
    data = data.dropna(subset=fitur)

    # Memilih data yang akan digunakan untuk analisis
    X = data[fitur].values

    # Inisialisasi model LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)

    # Memprediksi anomali
    anomali = lof.fit_predict(X)

    # Filter data yang merupakan anomali
    anomaly_data = data.iloc[np.where(anomali == -1)]

    # Export data yang berisi anomali ke dalam file Excel
    output_file = "/content/drive/My Drive/Colab Notebooks/Anomali pada Usia.xlsx"
    try:
        anomaly_data.to_excel(output_file, index=False)
        st.write("Data anomali berdasarkan usia berhasil diekspor ke:", output_file)
    except Exception as e:
        st.write("Gagal mengexport data:", str(e))

    return data, anomaly_data, X, anomali

# Fungsi untuk membaca data dari Excel dan memproses anomali berdasarkan kolom KET.
def process_anomalies_based_on_ket():
    data = pd.read_excel("/content/drive/My Drive/Colab Notebooks/Convert.xlsx")

    # Konversi kolom 'KELAMIN' menjadi numerik menggunakan LabelEncoder
    label_encoder = LabelEncoder()
    data['KELAMIN'] = label_encoder.fit_transform(data['KELAMIN'])

    # Pilih fitur untuk X dan y
    X = data[['USIA']].values  # label x
    y = data['KELAMIN'].values  # label y

    # Deteksi anomali berdasarkan kolom 'KET.'
    data['is_anomaly'] = data['KET.'].notna().astype(int)

    # Tangani nilai NaN dalam kolom 'USIA'
    X = np.nan_to_num(X)

    # Proyeksi data anomali
    anomalies = data[data['is_anomaly'] == 1]

    # Ubah kembali isi kolom 'KELAMIN' menjadi 0=L dan 1=P
    gender_mapping = {0: 'L', 1: 'P'}
    anomalies.loc[:, 'KELAMIN'] = anomalies['KELAMIN'].map(gender_mapping)

    # Export data yang berisi anomali ke dalam file Excel
    output_file = "/content/drive/My Drive/Colab Notebooks/Data Anomali.xlsx"
    try:
        anomalies.to_excel(output_file, index=False)
        st.write("Data Anomali berhasil diekspor ke:", output_file)
    except Exception as e:
        st.write("Gagal mengexport data:", str(e))

    return data, anomalies

# Tampilan Streamlit
def main():
    st.title("Deteksi Anomali")

    st.header("Anomali Berdasarkan Usia")
    data, anomaly_data, X, anomali = read_and_process_data()

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(X[:, 0], data['KELAMIN'], color='blue', label='Data Asli')
        ax.scatter(X[anomali == -1][:, 0], data['KELAMIN'].iloc[anomali == -1], color='red', label='Anomali')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Laki-laki', 'Perempuan'])
        ax.set_title('Deteksi Anomali Berdasarkan Usia')
        ax.set_xlabel('Usia')
        ax.set_ylabel('JENIS KELAMIN')
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.write("Data Anomali Berdasarkan Usia:")
        st.write(anomaly_data)

    st.header("Anomali Berdasarkan KET.")
    data, anomalies = process_anomalies_based_on_ket()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data['USIA'], data['KELAMIN'], color='blue', label='Normal Data')
        ax.scatter(anomalies['USIA'], anomalies['KELAMIN'], color='red', label='Anomalies')
        ax.set_title('Deteksi Keseluruhan Anomali')
        ax.set_xlabel('Usia')
        ax.set_ylabel('Kelamin')
        ax.legend()
        st.pyplot(fig)

    with col4:
        st.write("Data Anomali Berdasarkan KET.:")
        st.write(anomalies)

if __name__ == "__main__":
    main()
