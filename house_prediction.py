import streamlit as st
import numpy as np
import joblib

# Memuat model yang sudah dilatih (dalam hal ini, Support Vector Regressor)
model = joblib.load('linear_regression_model.pkl')

# Fungsi untuk memprediksi harga
def predict_price(square_feet, year_built, location, bedrooms, bathrooms):
    # Pengkodean sederhana untuk lokasi jika bersifat kategorikal (disesuaikan sesuai kebutuhan)
    if location == 'Urban':
        location_encoded = 0
    elif location == 'Suburban':
        location_encoded = 1
    else:
        location_encoded = 2  # Untuk 'Rural'
    
    # Mempersiapkan input untuk prediksi (disesuaikan dengan bagaimana model dilatih)
    features = np.array([[square_feet, year_built, location_encoded, bedrooms, bathrooms]])
    
    # Melakukan prediksi
    predicted_price = model.predict(features)
    
    return predicted_price

# Judul aplikasi Streamlit
st.title('Prediksi Harga Rumah')

# Kolom input untuk pengguna memasukkan data
square_feet = st.number_input('Masukkan Luas Rumah (Square Feet):', min_value=100, max_value=10000, step=10)
bedrooms = st.number_input('Masukkan Jumlah Kamar Tidur (Bedrooms):', min_value=1, max_value=10, step=1)
bathrooms = st.number_input('Masukkan Jumlah Kamar Mandi (Bathrooms):', min_value=1, max_value=10, step=1)
year_built = st.number_input('Masukkan Tahun Dibangun:', min_value=1800, max_value=2024, step=1)
location = st.selectbox('Pilih Lokasi Rumah:', ('Urban', 'Suburban', 'Rural'))

# Tombol untuk memprediksi
if st.button('Prediksi Harga'):
    price = predict_price(square_feet, year_built, location, bedrooms, bathrooms)
    st.write(f"Prediksi harga rumah: ${price[0]:,.2f}")

# Footer
st.write("Aplikasi prediksi harga rumah ini menggunakan model SVR.")
