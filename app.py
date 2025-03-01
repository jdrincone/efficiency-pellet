# ============================
# Importación de Librerías
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata



colors = ["#94AF92", "#1C8074"]

# Crear paleta personalizada usando LinearSegmentedColormap
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)


# ============================
# Configuración de la App
# ============================
st.set_page_config(page_title="📊 Tablero de Análisis de Durabilidad", layout="wide")

# ============================
# Funciones de Utilidad
# ============================
@st.cache_data
def cargar_datos(file_name):
    """Carga datos desde la carpeta data."""
    return pd.read_csv(f"data/{file_name}.csv")

def plot_general_hist(df):
    """Gráfico de Histograma de la Durabilidad por Peletizadora."""
    df["Peletizadora"] = df["Peletizadora"].astype(str)
    
    mean_value_1 = df[df["Peletizadora"] == '1']["Durabilidad"].mean()
    mean_value_2 = df[df["Peletizadora"] == '2']["Durabilidad"].mean()

    fig = px.histogram(
        df,
        x="Durabilidad",
        color="Peletizadora",
        opacity=0.8,
    )

    fig.add_vline(x=mean_value_1, line_dash="dash", line_color="#1C8074", 
                annotation_text=f"<span style='color:#1C8074;'><b>Media {mean_value_1:.2f}</b></span> ")

    fig.add_vline(x=mean_value_2, line_dash="dot", line_color="#1C8074", 
                annotation_text=f"<span style='color:#1C8074;'><b>Media {mean_value_2:.2f}</b></span>")

    fig.update_layout(
        xaxis_title="Durabilidad (%)",      # Etiqueta en eje X
        yaxis_title="Frecuencia",           # Etiqueta en eje Y
        #title="Distribución de la Durabilidad por Peletizadora",
        #title_x=0.5,                       # Centrado del título
        legend_title="Peletizadora"         # Título de la leyenda
    )

    st.plotly_chart(fig)


def summary_df(category, data):
    """Resumen Estadístico Agrupado."""
    summary = data.groupby(category)["Durabilidad"].agg(["median", "mean", "count", "std", "min", "max"]).reset_index()
    summary["Error Estandar"] = summary["std"] / summary["count"] ** 0.5
    summary["Limite Inf (95%)"] = summary["mean"] - 1.96 * summary["Error Estandar"]
    summary["Limite Super (95%)"] = summary["mean"] + 1.96 * summary["Error Estandar"]
    summary.rename(columns={"std": "Desviación Estandar", "count": "Cantidad Medidas",
                            "mean":"Valor Medio", "median": "Mediana", 
                            "min": "Mínimo", "max": "Máximo"}, inplace=True)
    return summary
# 📌 Color base y gradiente
base_color = "#1C8074"
light_color = "#A7D3C7"  # Un verde suave derivado del color base

# 🔹 Crear la paleta de colores personalizada
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [light_color, base_color])

# ============================
# Carga de Datos
# ============================
df = cargar_datos("data_convension")

# ============================
# Sidebar
# ============================
st.sidebar.image("images/logo.png", width=150)
st.sidebar.markdown("### Datos de calidad en peletizadoras de Cervalle")

# ============================
# Imagen Principal
# ============================
imagen = Image.open('images/logo_ppal.jpg')
st.image(imagen, use_container_width=True)

# ============================
# Título y Subtítulo
# ============================
st.markdown(f"""
    <h1 style="text-align: center; 
               font-size: 90px;
               background: black; 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;">
               Eficiencia en la durabilidad del Pellet
    </h1>
""", unsafe_allow_html=True)

st.markdown(f"""
    <h2 style="text-align: center; 
               background: #1C8074; 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;">
        El poder de los datos para inferir, pronosticar y optimizar procesos en los negocios
    </h2>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


st.markdown("""
    <div style="text-align: center; font-size: 24px;">
        <h3 style="font-size: 32px;">🔍 Punto de Partida</h3>
        <ul style="list-style-type: none; padding: 0; font-size: 24px;">
            <li>📏 <b>Datos:</b> Recolección, selección y limpieza de datos relevantes.</li>
            <li>🔬 <b>Contexto:</b> Personas, variables y factores considerados.</li>
            <li>🌍 <b>Objetivos:</b> Entendimiento/Estrategía del negocio ¿Qué queremos inferir, pronosticar y optimizar?</li>
        </ul>
    </div>
""", unsafe_allow_html=True)


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
    <h2 style="text-align: center; 
               background: black; 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;">
        🔬 Variables que influyen en la calidad del pellet 🔬
    </h2>
""", unsafe_allow_html=True)
#st.markdown("<br><br>", unsafe_allow_html=True)
#st.markdown("<br><br>", unsafe_allow_html=True)
image = Image.open('images/pel.png')  # Reemplaza con la ruta de tu imagen
#resized_image = image.resize((600, 600))
st.image(image=image)

st.markdown("<br><br>", unsafe_allow_html=True)

# ============================
# Gráficos y Análisis
# ============================
st.markdown("""
    <h3 style='text-align: center; margin-bottom: 0.4cm;'></h3>
""", unsafe_allow_html=True)

st.markdown(f"""
    <h2 style="text-align: center; 
               background: black; 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;">
               Generalidades de la Durabilidad
    </h2>
""", unsafe_allow_html=True)
sumary = summary_df("Peletizadora", df)

# ============================
# Mostrar Tabla Centrando
# ============================
st.markdown(
    """
    <style>
    .dataframe-container {
        display: flex;
        justify-content: center;
    }
    </style>
    <div class="dataframe-container">
    """, 
    unsafe_allow_html=True
)
st.dataframe(sumary)
st.markdown('</div>', unsafe_allow_html=True)


st.markdown("""
    <h3 style='text-align: center;'>Distribución de las medidas de durabilidad</h3>
""", unsafe_allow_html=True)
plot_general_hist(df)






st.markdown("""
    <h3 style='text-align: center;'>Matriz de correlacción Peletizadora 2</h3>
""", unsafe_allow_html=True)

image = Image.open('images/correlation_heatmap.png')
st.image(image)



st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

# ============================
# Imagen Inferencia y Pronóstico
# ============================
st.markdown(f"""
    <h2 style="text-align: center; 
               background: #1C8074; 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;">
        Inferencia y prónostico
    </h2>
""", unsafe_allow_html=True)

image = Image.open('images/ML.png')
st.image(image.resize((1200, 600)))



st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)




st.markdown("""
    <h3 style='text-align: center;'>Explicabilidad del modelo</h3>
""", unsafe_allow_html=True)

image = Image.open('images/shap_top5_bar.png')
st.image(image)



st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


st.markdown("""
    <h3 style='text-align: center;'>Comportamiento de cada variable en la predicción</h3>
""", unsafe_allow_html=True)

image = Image.open('images/shap_scatter_Delta Temp.png')
st.image(image)

image = Image.open('images/shap_scatter_Hum Enfriamiento.png')
st.image(image)



st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(f"""
    <h2 style="text-align: center; 
               background: #1C8074; 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;">
        Simulación
    </h2>
""", unsafe_allow_html=True)

st.markdown(f"""
    <h2 style="text-align: center; 
               background: black; 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;">
        Explorar espacios de parámetros del sistema ¿Qué pasará si...?
    </h2>
""", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata
import matplotlib.colors as mcolors


@st.cache_data
def load_result():
    return  pd.read_csv("data/df_results_conv.csv")
df_results = load_result()
# 📌 Extraer los datos del DataFrame
x = df_results["Hum Enfriamiento"]
y = df_results["Temp Enfriamiento"]
z = df_results["Durabilidad media"]

# 📌 Crear dos columnas en Streamlit para mostrar ambas gráficas en la misma página
col1, col2 = st.columns(2)



# 📌 Convertir la paleta de Matplotlib a una escala compatible con Plotly
plotly_colorscale = [[i, mcolors.to_hex(custom_cmap(i))] for i in np.linspace(0, 1, 100)]


# 📊 **Gráfico 3D en la primera columna (Plotly)**
with col1:
    

    fig_3d = px.scatter_3d(
        df_results, x="Hum Enfriamiento", y="Temp Enfriamiento", z="Durabilidad media",
        color="Durabilidad media", 
        color_continuous_scale=plotly_colorscale,
        opacity=0.8
    )

    fig_3d.update_layout(
        width=500, height=500,  # Ajustar tamaño
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis_title="Hum Enfriamiento",
            yaxis_title="Temp Enfriamiento",
            zaxis_title="Durabilidad Media"
        )
    )

    st.plotly_chart(fig_3d, use_container_width=True)

# 📊 **Gráfico de Contornos en la segunda columna (Plotly)**
with col2:

    # 📌 Crear un grid para la interpolación
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)

    # 📌 Interpolar los valores de Z (Durabilidad Media)
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

    # 📊 Crear el gráfico de contornos con la paleta personalizada
    fig_contour = go.Figure(data=go.Contour(
        x=xi, y=yi, z=Zi,
        colorscale=plotly_colorscale,  # 🔥 Aplicar la paleta personalizada
        contours=dict(start=z.min(), end=z.max(), size=(z.max()-z.min())/20)
    ))

    fig_contour.update_layout(
        width=500, height=500,  # Ajustar tamaño
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="Hum Enfriamiento",
        yaxis_title="Temp Enfriamiento",
    )

    st.plotly_chart(fig_contour, use_container_width=True)


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)


st.markdown("""
    <div style="text-align: center; font-size: 24px;">
        <h3 style="font-size: 32px;">Apenas iniciamos la aventura en los datos...</h3>
        <ul style="list-style-type: none; padding: 0; font-size: 24px;">
            <li> - Entender al detalle los datos, desde el proceso en produccción con la 
                   toma de medidas hasta la estrategía del negocio.</li>
            <li> - ¿Qué esta mal? Corregir y Aprender de ello ¿Qué esta bien? Seguir realizandolo.</li>
            <li> - Experimentar, Experimentar  y Experimentar, iterar para refinar  y aprender de los datos. </li>
            <li> - Soñar en grande con los pies en la tierra, domesticar  mostros con técnica y cerebro. </li>
        </ul>
    </div>
""", unsafe_allow_html=True)