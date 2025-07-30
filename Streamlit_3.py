import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

st.set_page_config(page_title="Galaxy Classifier", layout="wide")
st.title("Galaxy Barred/Unbarred Classification")

# Initialize session state for galaxy index
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# File uploader
if 'df' not in st.session_state:
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        for col in ['image_url', 'model_url', 'residual_url', 'classification']:
            if col not in st.session_state.df.columns:
                st.session_state.df[col] = ''
else:
    uploaded_file = None

st.sidebar.header("Settings")
cutout_base = 'https://www.legacysurvey.org/viewer/jpeg-cutout'
zoom = st.sidebar.slider("Zoom level", min_value=5, max_value=20, value=10)
size = st.sidebar.slider("Image size (px)", min_value=128, max_value=512, step=64, value=256)

layers = {
    'Data': 'ls-dr10',
    'Model': 'ls-dr10-model',
    'Residual': 'ls-dr10-resid'
}

stretch_type = st.sidebar.selectbox("Image Stretch:", ['none', 'log', 'asinh'], index=1)
show_crop = st.sidebar.checkbox("Show Center Zoom-In", value=False)

cmap_choice = st.sidebar.selectbox("Color Map for Data/Model", ['gray', 'viridis', 'hot', 'cool', 'plasma', 'magma', 'cividis'])

def fetch_and_process(url, stretch):
    r = requests.get(url)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert('L')
    arr = np.array(img)
    if stretch == 'log':
        arr = np.log1p(arr)
    elif stretch == 'asinh':
        arr = np.arcsinh(arr)
    arr = exposure.rescale_intensity(arr, out_range=(0,255)).astype(np.uint8)
    return arr

if 'df' in st.session_state:
    df = st.session_state.df
    total = len(df)

    nav1, nav2, nav3 = st.columns([1,2,1])
    with nav1:
        if st.button("Previous Galaxy") and st.session_state.idx > 0:
            st.session_state.idx -= 1
    with nav2:
        st.markdown(f"### Galaxy {st.session_state.idx+1} of {total}")
    with nav3:
        if st.button("Next Galaxy") and st.session_state.idx < total-1:
            st.session_state.idx += 1

    row = df.iloc[st.session_state.idx]
    ra, dec = row['RAJ2000'], row['DEJ2000']
    urls = {name: f"{cutout_base}?ra={ra}&dec={dec}&layer={layer}&zoom={zoom}&size={size}"
            for name, layer in layers.items()}

    cols = st.columns(3)
    for col_box, (name, url) in zip(cols, urls.items()):
        with col_box:
            st.markdown(f"**{name}**")
            try:
                arr = fetch_and_process(url, stretch_type)
                fig, ax = plt.subplots(figsize=(3,3))
                cmap = 'seismic' if name == 'Residual' else cmap_choice
                ax.imshow(arr, origin='lower', cmap=cmap)
                ax.axis('off')
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                st.image(buf.getvalue(), use_column_width=True)

                if show_crop:
                    h, w = arr.shape
                    crop = arr[h//4:3*h//4, w//4:3*w//4]
                    fig2, ax2 = plt.subplots(figsize=(3,3))
                    ax2.imshow(crop, origin='lower', cmap=cmap)
                    ax2.axis('off')
                    buf2 = BytesIO()
                    fig2.savefig(buf2, format='png', bbox_inches='tight', pad_inches=0)
                    plt.close(fig2)
                    st.image(buf2.getvalue(), caption='Center Zoom', use_column_width=True)

            except Exception as e:
                st.error(f"Error loading {name}: {e}")

    options = ["barred", "unbarred", "Not Sure"]
    current = df.at[st.session_state.idx, 'classification']
    default_idx = options.index(current) if current in options else 0
    label = st.radio(
        f"Classify galaxy {row.get('Name_x','')} (RA={ra}, Dec={dec}):",
        options=options,
        index=default_idx,
        key=f"class_{st.session_state.idx}"
    )

    df.at[st.session_state.idx, 'image_url'] = urls['Data']
    df.at[st.session_state.idx, 'model_url'] = urls['Model']
    df.at[st.session_state.idx, 'residual_url'] = urls['Residual']
    df.at[st.session_state.idx, 'classification'] = label

    df.to_csv("autosave_classified.csv", index=False)

    st.markdown("---")
    st.markdown("💾 **Download your current classification data**")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Classified CSV",
        data=csv,
        file_name="galaxies_classified.csv",
        mime='text/csv'
    )

else:
    st.info("Please upload a CSV file to begin classification.")
