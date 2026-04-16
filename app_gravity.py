import streamlit as st
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import harmonica as hm
import tempfile
import os
import plotly.graph_objects as go

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(page_title="Modul Koreksi Gravity", layout="wide")
st.title("Grav-inv3D: FAA to CBA Converter (Prism Method)")

# --- FUNGSI VISUALISASI ALA SURFER ---
def plot_smooth_grid(x, y, z_matrix, title, colorscale='Jet'):
    """Fungsi untuk membuat peta kontur mulus ala Surfer menggunakan Plotly"""
    fig = go.Figure(data=go.Contour(
        z=z_matrix,
        x=x, # Koordinat X (1D array)
        y=y, # Koordinat Y (1D array)
        colorscale=colorscale,
        contours=dict(showlines=False), # Menghilangkan garis kontur kasar, menjadikannya smooth grid
        colorbar=dict(title="mGal")
    ))
    
    fig.update_layout(
        title=title,
        autosize=False,
        width=700,
        height=600,
        margin=dict(l=50, r=50, b=50, t=50)
    )
    # Memastikan aspek rasio peta proporsional (tidak distorsi)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig

# --- FUNGSI PEMROSESAN UTAMA ---
def process_faa_to_cba_prism(faa_file_path, elev_file_path):
    # 1. Membaca data grid
    faa_ds = rioxarray.open_rasterio(faa_file_path, parse_coordinates=True)
    elev_ds = rioxarray.open_rasterio(elev_file_path, parse_coordinates=True)
    
    if faa_ds.rio.crs is None:
        faa_ds.rio.write_crs("EPSG:32748", inplace=True) 
    if elev_ds.rio.crs is None:
        elev_ds.rio.write_crs("EPSG:32748", inplace=True)

    # 2. Resampling & Ekstraksi Matriks
    elev_matched = elev_ds.rio.reproject_match(faa_ds)
    
    # --- KODE TAMBAHAN UNTUK MEMPERBAIKI ERROR PRISMA ---
    # Mengurutkan koordinat agar X naik (Barat -> Timur) dan Y naik (Selatan -> Utara)
    faa_ds = faa_ds.sortby(['x', 'y'])
    elev_matched = elev_matched.sortby(['x', 'y'])
    # ----------------------------------------------------
    
    # Baru kita ekstrak nilainya
    faa_val = faa_ds.values[0]
    z_val = elev_matched.values[0]
    
    x_coords = faa_ds.x.values
    y_coords = faa_ds.y.values
    X, Y = np.meshgrid(x_coords, y_coords)

    # 3. Model Prisma 3D
    rho_darat = 2670.0  
    rho_laut = 1030.0   
    density_grid = np.where(z_val >= 0, rho_darat, (rho_darat - rho_laut))
    
    prisms = hm.prism_layer(
        coordinates=(x_coords, y_coords),
        surface=z_val,
        reference=0,
        properties={"density": density_grid}
    )

    # 4. Hitung Efek Topografi Total & CBA
    Z_obs = np.full_like(X, 1.0) 
    topo_effect = prisms.prism_layer.gravity(coordinates=(X, Y, Z_obs), field="g_z")
    cba_val = faa_val - topo_effect

    # 5. Susun DataFrame untuk Export CSV
    df = pd.DataFrame({
        "X_UTM": X.ravel(),
        "Y_UTM": Y.ravel(),
        "Elevasi_m": z_val.ravel(),
        "FAA_mGal": faa_val.ravel(),
        "CBA_mGal": cba_val.ravel()
    }).dropna()
    
    # --- TAMBAHAN UNTUK MEMPERBAIKI WINERROR 32 ---
    # Memaksa Python melepas kunci file dari memory
    faa_ds.close()
    elev_ds.close()
    elev_matched.close()
    # -----------------------------------------------
    
    # Mengembalikan dataframe dan array 2D untuk kebutuhan plotting
    return df, x_coords, y_coords, faa_val, cba_val

# --- ANTARMUKA UI STREAMLIT ---
st.sidebar.header("📂 Input Data (.grd UTM)")
faa_file = st.sidebar.file_uploader("Upload Grid FAA", type=['grd'])
elev_file = st.sidebar.file_uploader("Upload Grid Elevasi", type=['grd'])

if st.sidebar.button("⚙️ Mulai Proses Koreksi"):
    if faa_file and elev_file:
        with st.spinner("Sedang memproses Resampling & Komputasi Prisma 3D... Mohon tunggu, ini mungkin memakan waktu sesaat."):
            try:
                # Simpan file dari memory ke temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".grd") as tmp_faa:
                    tmp_faa.write(faa_file.getvalue())
                    faa_path = tmp_faa.name
                    
                with tempfile.NamedTemporaryFile(delete=False, suffix=".grd") as tmp_elev:
                    tmp_elev.write(elev_file.getvalue())
                    elev_path = tmp_elev.name

                # Eksekusi Pemrosesan
                hasil_df, x_arr, y_arr, faa_matrix, cba_matrix = process_faa_to_cba_prism(faa_path, elev_path)
                
                # Biarkan Windows yang mengurus file temporary-nya nanti
                # os.remove(faa_path)  <-- HAPUS / BERI TANDA PAGAR BARIS INI
                # os.remove(elev_path) <-- HAPUS / BERI TANDA PAGAR BARIS INI

                st.success("✅ Koreksi berhasil diselesaikan!")
                
                # --- MENAMPILKAN VISUALISASI ---
                st.subheader("📊 Visualisasi Peta Anomali")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Free Air Anomaly (Input)**")
                    fig_faa = plot_smooth_grid(x_arr, y_arr, faa_matrix, "FAA (mGal)", colorscale='Jet')
                    st.plotly_chart(fig_faa, use_container_width=True)
                    
                with col2:
                    st.markdown("**Complete Bouguer Anomaly (Hasil Koreksi)**")
                    fig_cba = plot_smooth_grid(x_arr, y_arr, cba_matrix, "CBA (mGal)", colorscale='Jet')
                    st.plotly_chart(fig_cba, use_container_width=True)
                
                # --- TOMBOL DOWNLOAD & PREVIEW DATA ---
                st.subheader("💾 Export Data Hasil Koreksi")
                st.dataframe(hasil_df.head(10)) # Tampilkan 10 baris pertama sebagai preview
                
                csv = hasil_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data CBA (Format CSV)",
                    data=csv,
                    file_name='Data_CBA_Prism.csv',
                    mime='text/csv',
                )
                
            except Exception as e:
                st.error(f"Terjadi kesalahan komputasi: {e}")
    else:
        st.warning("⚠️ Mohon upload kedua file grid terlebih dahulu di sidebar sebelah kiri.")
else:
    st.info("Silakan unggah file grid FAA dan Elevasi, lalu klik tombol 'Mulai Proses Koreksi'.")