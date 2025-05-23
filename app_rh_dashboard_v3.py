
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import numpy as np
from PIL import Image
from io import BytesIO
from dateutil.relativedelta import relativedelta
import unicodedata
import json
import plotly.express as px
import matplotlib.dates as mdates
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans





st.set_page_config(page_title="Decision Dashboard RH", layout="wide")
sns.set(style="whitegrid")

st.markdown("""
<style>
.sidebar .stSelectbox, .sidebar .stDateInput, .sidebar .stTextInput {
    margin-bottom: 6px !important;
    padding: 4px 6px !important;
}
.element-container input[type="text"], .stSelectbox div[data-baseweb="select"] {
    font-size: 14px !important;
    height: 28px !important;
}
.stSelectbox > label, .stTextInput > label, .stDateInput > label {
    font-size: 12px !important;
    margin-bottom: 2px !important;
}
.stSidebar > div:first-child {
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}
hr {
    margin: 5px 0 !important;
    border: none;
    border-top: 1px solid #ccc;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 📂 Navegação", unsafe_allow_html=True)
    aba = st.radio(
        "Selecione uma seção:",
        ["📊 Dashboards", "🧑‍💼 Previsões"],
        label_visibility="collapsed"
    )
st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)


if aba == "📊 Dashboards":
    #st.title("📊 Painel de Vagas")
    st.markdown("<h1 style='text-align: center;'>📊 Decision Dashboards RH</h1>", unsafe_allow_html=True)
    with open("vagas.json", encoding="utf-8") as f:
        json_data = pd.read_json(f)

    df_completo_exportado = pd.read_csv ("df_completo_exportado.csv")

    st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

    # Converter campo de data
    df_completo_exportado["data_requicisao"] = pd.to_datetime(df_completo_exportado["data_requicisao"], errors="coerce")

    # Definir limites
    min_data = df_completo_exportado["data_requicisao"].min().date()
    max_data = df_completo_exportado["data_requicisao"].max().date()
    default_start = max_data - relativedelta(months=3)

    with st.sidebar:
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>🔍 Filtros</h3>", unsafe_allow_html=True)
        st.markdown("### 🗓️ Período", unsafe_allow_html=True)

        # Campos de data
        data_inicio = st.date_input(
            "Data Inicial",
            value=default_start,
            min_value=min_data,
            max_value=max_data,
            key="data_inicio"
        )
        data_fim = st.date_input(
            "Data Final",
            value=max_data,
            min_value=min_data,
            max_value=max_data,
            key="data_fim"
        )

        # Validação lógica
        if data_inicio > data_fim:
            st.error("❌ A data inicial não pode ser maior que a data final.")
            st.stop()
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)



        st.markdown("### 🌐 Idiomas")
        # Filtro de Nível de Inglês
        if "filtro_ingles" not in st.session_state or st.session_state.get("reset", False):
            st.session_state["filtro_ingles"] = "Todos"
        filtro_ingles = st.selectbox(
            "Nível de Inglês",
            options=["Todos"] + sorted(df_completo_exportado["nivel_ingles_vaga"].dropna().unique()),
            key="filtro_ingles"
        )
        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)



        st.markdown("### 📌 Outros Filtros", unsafe_allow_html=True)
# Cliente
        if "filtro_cliente" not in st.session_state or st.session_state.get("reset", False):
            st.session_state["filtro_cliente"] = "Todos"
        filtro_cliente = st.selectbox(
            "Cliente",
            options=["Todos"] + sorted(df_completo_exportado["cliente"].dropna().unique()),
            key="filtro_cliente"
        )



# Estado
        if "filtro_estado" not in st.session_state or st.session_state.get("reset", False):
            st.session_state["filtro_estado"] = "Todos"
        filtro_estado = st.selectbox(
            "Estado",
            options=["Todos"] + sorted(df_completo_exportado["estado"].dropna().unique()),
            key="filtro_estado"
        )

        st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)
        st.markdown("### 🧾 Tipo de Vaga", unsafe_allow_html=True)

       


# Nível Profissional
        if "filtro_nivel" not in st.session_state or st.session_state.get("reset", False):
            st.session_state["filtro_nivel"] = "Todos"
        filtro_nivel = st.selectbox(
            "Nível Profissional",
            options=["Todos"] + sorted(df_completo_exportado["nivel profissional"].dropna().unique()),
            key="filtro_nivel"
        )


       # Espaço e separador
        st.markdown("<hr style='margin: 10px 5px;'>", unsafe_allow_html=True)

        # Botão centralizado com CSS
        st.markdown(
            """
            <div style='display: flex; justify-content: center; margin-top: 10px;'>
                <form action="">
                    <button type="submit" style="
                        background-color: #e6e6f2;
                        color: #182173;
                        border: 1px solid #333382;
                        padding: 0.5rem 1rem;
                        font-size: 14px;
                        border-radius: 5px;
                        cursor: pointer;
                    ">
                        🔄 Limpar Filtros
                    </button>
                </form>
            </div>
            """,
            unsafe_allow_html=True
        )        


    df_filtrado = df_completo_exportado[
        (df_completo_exportado["data_requicisao"].dt.date >= data_inicio) &
        (df_completo_exportado["data_requicisao"].dt.date <= data_fim)
    ]
    if filtro_ingles != "Todos":
        df_filtrado = df_filtrado[df_filtrado["nivel_ingles"] == filtro_ingles]
    if filtro_nivel != "Todos":
        df_filtrado = df_filtrado[df_filtrado["nivel profissional"] == filtro_nivel]
    if filtro_cliente != "Todos":
        df_filtrado = df_filtrado[df_filtrado["cliente"] == filtro_cliente]
    if filtro_estado != "Todos":
        df_filtrado = df_filtrado[df_filtrado["estado"] == filtro_estado]



    st.markdown("""
    <div style='text-align: center; font-size: 24px; margin-bottom: 20px; font-weight: bold;'>
        📈 Indicadores
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.05);'>
                <h4 style='margin-bottom: 5px;'>📋 Vagas</h4>
                <p style='font-size: 28px; font-weight: bold; color: #2c3e50;'>{len(df_filtrado)}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.05);'>
                <h4 style='margin-bottom: 5px;'>👥 Clientes</h4>
                <p style='font-size: 28px; font-weight: bold; color: #2c3e50;'>{df_filtrado["cliente"].nunique()}</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        df_sap = df_filtrado[df_filtrado["titulo_vaga"].str.contains("sap", case=False, na=False)]
        num_sap = df_sap.shape[0]

        st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.05);'>
                <h4 style='margin-bottom: 5px;'>🛠️ Vagas SAP Encontradas</h4>
                <p style='font-size: 28px; font-weight: bold; color: #2c3e50;'>{num_sap}</p>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<hr style='margin: 10px 5px;'>", unsafe_allow_html=True)



    col1, col2 = st.columns(2)

    # 📊 Top 10 Títulos de Vagas (barras verticais)
    with col1:
        st.markdown("<h3 style='text-align: center; color:#0a0d2b;'>📊 Top 10 Vagas</h3>", unsafe_allow_html=True)
        top_vagas = df_filtrado["titulo_vaga"].value_counts().head(10).sort_values(ascending=True)

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.barplot(x=top_vagas.index, y=top_vagas.values, ax=ax1, palette="Blues")

        for i, v in enumerate(top_vagas.values):
            ax1.text(i, v + 0.3, str(v), ha='center', fontsize=8, fontweight='bold')

        ax1.set_xlabel("")
        ax1.set_ylabel("")
        ax1.set_yticks([])
        ax1.tick_params(axis='x', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        sns.despine(left=True, bottom=True)
        st.pyplot(fig1)

    with col2:
        st.markdown("<h3 style='text-align: center; color:#0a0d2b;'>🌐 Vagas por Estado</h3>", unsafe_allow_html=True)

    # Contagem por estado
        vagas_estado = df_filtrado["estado"].value_counts().head(10).sort_values(ascending=True)
    
    # Estrutura para heatmap
        heat_data = pd.DataFrame(vagas_estado).reset_index()
        heat_data.columns = ["estado", "quantidade"]
        heat_matrix = heat_data.pivot_table(index="estado", values="quantidade").astype(int)

    # Criação do heatmap com tons azuis modernos
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            heat_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",  # moderno e suave em azul
            linewidths=0.5,
            linecolor="#e0e0e0",
            cbar=False,
            ax=ax4
        )

        ax4.set_xlabel("")
        ax4.set_ylabel("")
        #ax4.tick_params(axis='y', labelsize=9)
        sns.despine(left=True, bottom=True)
        st.pyplot(fig4)


    st.markdown("<hr style='margin: 10px 5px;'>", unsafe_allow_html=True)

    # Agrupando as vagas por data (já filtrado pelo seu filtro de datas)
    df_time_grouped = df_filtrado.groupby("data_requicisao").size().reset_index(name="quantidade")

    # Gráfico interativo com tooltip
    fig = px.line(
        df_time_grouped,
        x="data_requicisao",
        y="quantidade",
        markers=True,
        title="📅 Evolução de Vagas por Data de Requisição",
        labels={"data_requicisao": "Data", "quantidade": "Quantidade de Vagas"},
        template="plotly_white",
        color_discrete_sequence=["#1f77b4"]
    )

    # Estilo visual moderno
    fig.update_traces(
        hovertemplate="Data: %{x}<br>Qtd: %{y}",
        line=dict(width=3),
        marker=dict(size=6)
    )
    fig.update_layout(
        title_x=0.5,
        hovermode="x unified",
        margin=dict(t=50, b=40, l=40, r=40)
    )

    # Renderiza no Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr style='margin: 10px 5px;'>", unsafe_allow_html=True)


    st.markdown("<h3 style='text-align: center; color:#0a0d2b;'>🏷️ Nível Profissional</h3>", unsafe_allow_html=True)

    nivel_counts = df_filtrado["nivel profissional"].value_counts().head(10).sort_values(ascending=False)

    fig_nv2, ax_nv2 = plt.subplots(figsize=(7, 4))
    sns.barplot(y=nivel_counts.index, x=nivel_counts.values, ax=ax_nv2, palette="Blues_r")

    for i, v in enumerate(nivel_counts.values):
        ax_nv2.text(v + 0.3, i, str(v), va='center', fontsize=8, fontweight='bold')

    ax_nv2.set_xlabel("")
    ax_nv2.set_ylabel("")
    ax_nv2.set_xticks([])
    sns.despine(left=True, bottom=True)
    st.pyplot(fig_nv2)

    st.markdown("<hr style='margin: 10px 5px;'>", unsafe_allow_html=True)
   


   # 📊 Top 10 Recrutadores - Barras verticais (ordenado do menor para maior)
    top_recrutadores = df_filtrado["recrutador"].value_counts().head(10).sort_values(ascending=True)

    # Criação do gráfico com estilo moderno
    fig4, ax4 = plt.subplots(figsize=(7, 4))
    sns.barplot(x=top_recrutadores.index, y=top_recrutadores.values, ax=ax4, palette="Blues_r")

    # Rótulos nas barras
    for i, v in enumerate(top_recrutadores.values):
        ax4.text(i, v + 0.3, str(v), ha='center', fontsize=8, fontweight='bold')

    # Estilo visual
    ax4.set_xlabel("")
    ax4.set_ylabel("")
    ax4.set_yticks([])
    ax4.tick_params(axis='x', labelsize=8)
    plt.xticks(rotation=45, ha='right')
    sns.despine(left=True, bottom=True)

    # Título estilizado
    #st.markdown("<h3 style='text-align: center; color:#0a0d2b;'>👩‍💼 Top 10 Recrutadores</h3>", unsafe_allow_html=True)
    st.pyplot(fig4)


    st.markdown("<hr style='margin: 10px 5px;'>", unsafe_allow_html=True)

    sap_counts = df_filtrado["vaga_sap"].value_counts().head(10).reset_index()
    sap_counts.columns = ["tipo", "quantidade"]
    fig2 = px.pie(sap_counts, names="tipo", values="quantidade", hole=0.4,
              title="Distribuição de Vagas SAP",
              color_discrete_sequence=px.colors.sequential.Blues_r)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr style='margin: 10px 5px;'>", unsafe_allow_html=True)

    colunas_desejadas = [
        "titulo_profissional",
        "nivel profissional",
        "vaga_especifica_para_pcd",
        "cidade",
        "estado",
        "tipo_contratacao",
        "cliente",
        "vaga_sap",
        "titulo_vaga",
        "situacao_candidado",
        "aderente",
        "nivel_ingles_vaga"
    ]

    df_modelo = df_filtrado[colunas_desejadas]


    #st.write("Colunas disponíveis:", df_modelo.columns.tolist())

    # Simulando os dados finais com base no seu esquema
    #data = [
    #    {"titulo_vaga": "analista sap", "titulo_profissional": "analista", "cliente": "cliente b", "nivel_profissional": "senior", "tipo_contratacao": "clt", "vaga_sap": "sim", "vaga_especifica_para_pcd": "nao", "nivel_ingles": "avancado", "aderente": 1},
    #    {"titulo_vaga": "dev sap", "titulo_profissional": "dev", "cliente": "cliente a", "nivel_profissional": "pleno", "tipo_contratacao": "pj", "vaga_sap": "sim", "vaga_especifica_para_pcd": "nao", "nivel_ingles": "intermediario", "aderente": 1},
    #    {"titulo_vaga": "arquiteto sap", "titulo_profissional": "arquiteto", "cliente": "cliente c", "nivel_profissional": "senior", "tipo_contratacao": "clt", "vaga_sap": "sim", "vaga_especifica_para_pcd": "sim", "nivel_ingles": "avancado", "aderente": 1},
    #    {"titulo_vaga": "consultor java", "titulo_profissional": "consultor", "cliente": "cliente a", "nivel_profissional": "junior", "tipo_contratacao": "pj", "vaga_sap": "nao", "vaga_especifica_para_pcd": "nao", "nivel_ingles": "basico", "aderente": 0},
    #    {"titulo_vaga": "qa tester", "titulo_profissional": "qa", "cliente": "cliente a", "nivel_profissional": "junior", "tipo_contratacao": "clt", "vaga_sap": "nao", "vaga_especifica_para_pcd": "sim", "nivel_ingles": "basico", "aderente": 0},
    #]

    #df_modelo = pd.DataFrame(df_modelo)


    # Recursos e alvo
    features = [
        "titulo_profissional", "vaga_especifica_para_pcd",
        "tipo_contratacao", "cliente", "vaga_sap", "titulo_vaga"
    ]
    target = "aderente"

    X = df_modelo[features]
    y = df_modelo[target]

    # Pré-processamento
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), features)
    ])

    # Pipeline
    model = Pipeline([
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Treino
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model.fit(X_train, y_train)

    # Streamlit
    st.title("🧠 Previsão de Aderência do Candidato")

    # Interface
    input_data = {}
    col1, col2 = st.columns(2)
    with col1:
        input_data["titulo_vaga"] = st.selectbox("Título da Vaga", df_modelo["titulo_vaga"].unique())
        input_data["cliente"] = st.selectbox("Cliente", df_modelo["cliente"].unique())
        input_data["vaga_sap"] = st.selectbox("Vaga SAP?", ["sim", "nao"])
        input_data["nivel_ingles"] = st.selectbox("Nível de Inglês", df_modelo["nivel_ingles_vaga"].unique())
    with col2:
        input_data["titulo_profissional"] = st.selectbox("Título Profissional", df_modelo["titulo_profissional"].unique())
        input_data["nivel_profissional"] = st.selectbox("Nível Profissional", df_modelo["nivel profissional"].unique())
        input_data["tipo_contratacao"] = st.selectbox("Tipo de Contratação", df_modelo["tipo_contratacao"].unique())
        input_data["vaga_especifica_para_pcd"] = st.selectbox("PCD?", ["sim", "nao"])

    if st.button("🔍 Prever Aderência"):
        input_df = pd.DataFrame([input_data])
        proba = model.predict_proba(input_df)[0][1]
        #st.metric("Probabilidade de Aderência", f"{proba*100:.1f}%")

    # Ordena os top 5 mais prováveis (simulando com df inteiro)
        full_proba = model.predict_proba(df_modelo[features])[:, 1]
        df_result = df_modelo.copy()
        df_result["proba"] = full_proba
        top_5 = df_result.sort_values("proba", ascending=False).head(5)[[
            "titulo_vaga", "titulo_profissional", "cliente", "proba"
        ]]

        st.markdown("### 🎯 Top 5 Candidatos Mais Aderentes")
        st.dataframe(top_5.rename(columns={"proba": "Probabilidade"}).assign(
            Probabilidade=lambda x: (x["Probabilidade"] * 100).round(1).astype(str) + "%"
        ))



if aba == "🧑‍💼 Previsões":
    st.title("🧑‍💼 Previsões de Candidatos")

    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics.pairwise import cosine_similarity

    #st.set_page_config(page_title="🧠 Match Candidato Ideal", layout="wide")

    df_completo_exportado = pd.read_csv ("df_completo_exportado.csv")

    colunas = [
    "titulo_vaga", "nivel profissional", "nivel_academico",
    "tipo_contratacao", "vaga_sap", "vaga_especifica_para_pcd", 
    "cliente", "estado", "cidade", "aderente", "nome_y", "nivel_ingles"
]
    
    

    #df_modelo_filtrado = df_completo_exportado[colunas].copy()

    #st.write("Colunas disponíveis:", df_completo_exportado.columns.tolist())


    st.markdown("""
        <style>
            .stSelectbox > div {
                font-size: 14px;
            }
        </style>
    """, unsafe_allow_html=True)

#st.title("🧠 Previsão de Aderência do Candidato Ideal")


# 🔢 FEATURES USADAS
    features = [
        "titulo_vaga", "nivel profissional", "nivel_academico",
        "tipo_contratacao", "vaga_sap", "vaga_especifica_para_pcd", 
        "cliente", "estado", "cidade"
    ]

# ⬇️ ENTRADA DO USUÁRIO
    st.markdown("### 🎯 Selecione os critérios desejados:")
    col1, col2 = st.columns(2)

    with col1:
        titulo_vaga = st.selectbox("Título da Vaga", df_completo_exportado["titulo_vaga"].dropna().unique())
        nivel_profissional = st.selectbox("Nível Profissional", df_completo_exportado["nivel profissional"].dropna().unique())
        nivel_academico = st.selectbox("Nível Acadêmico", df_completo_exportado["nivel_academico"].dropna().unique())
        tipo_contratacao = st.selectbox("Tipo de Contratação", df_completo_exportado["tipo_contratacao"].dropna().unique())
        nivel_ingles = st.selectbox("Nivel de Ingles", df_completo_exportado["nivel_ingles_vaga"].dropna().unique())
    with col2:
        vaga_sap = st.selectbox("É vaga SAP?", ["sim", "nao"])
        vaga_pcd = st.selectbox("É PCD?", ["sim", "nao"])
        cliente = st.selectbox("Cliente", df_completo_exportado["cliente"].dropna().unique())
        estado = st.selectbox("Estado", df_completo_exportado["estado"].dropna().unique())
        cidade = st.selectbox("Cidade", df_completo_exportado["cidade"].dropna().unique())

# 🔁 Quando clicar no botão
    if st.button("🔍 Buscar Candidatos Ideais"):

        

    # 1. Filtrar candidatos aderentes
        df_aderentes = df_completo_exportado[df_completo_exportado["aderente"] == 1].copy()

        df_aderentes.rename(columns={"nome_y": "nome_candidato"}, inplace=True)

    # 2. Criar base para modelar
        df_modelo = df_aderentes[features].fillna("nao_informado")

    # 3. Preprocessador
        cat_cols = df_modelo.select_dtypes(include="object").columns.tolist()
        preprocessor = ColumnTransformer([
            ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

    # 4. Fit e transformar os dados de candidatos
        X_transformed = preprocessor.fit_transform(df_modelo)

    # 5. Montar o vetor de entrada do usuário
        user_input = pd.DataFrame([{
            "titulo_vaga": titulo_vaga,
            "nivel profissional": nivel_profissional,
            "nivel_academico": nivel_academico,
            "tipo_contratacao": tipo_contratacao,
            "vaga_sap": vaga_sap,
            "vaga_especifica_para_pcd": vaga_pcd,
            "cliente": cliente,
            "estado": estado,
            "cidade": cidade
        }])

        user_vector = preprocessor.transform(user_input)  # matriz 1xN já
        sims = cosine_similarity(user_vector, X_transformed)[0]

        df_aderentes["similaridade"] = sims
        top_5 = df_aderentes.sort_values("similaridade", ascending=False).head(5).copy()
        media_sim = round(top_5["similaridade"].mean() * 100, 1)

        # 📊 Exibir
        st.markdown(f"### ✅ Aderência Média: `{media_sim}%`")
        st.dataframe(
            top_5[["titulo_vaga", "nome_candidato", "cliente", "estado", "similaridade"]]
            .rename(columns={"similaridade": "Aderência (%)"})
            .assign(**{"Aderência (%)": lambda df: (df["Aderência (%)"] * 100).round(1)})
            .style.format({"Aderência (%)": "{:.1f}%"})
        )



    
    

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import streamlit as st

    # 1. Features e Target
    features = [
        "titulo_vaga", "titulo_profissional", "nivel profissional",
        "nivel_academico", "tipo_contratacao", "vaga_sap",
        "vaga_especifica_para_pcd", "cliente", "estado", "cidade"
    ]

    target = "aderente"

    #df_aderentes.rename(columns={"nome_y": "nome_candidato"}, inplace=True)

    # 2. Separar dados aderentes (para treino)
    df_aderentes = df_completo_exportado[df_completo_exportado["aderente"].isin([0, 1])].copy()
    df_aderentes.rename(columns={"nome_y": "nome_candidato"}, inplace=True)
    df_aderentes = df_aderentes.dropna(subset=features)

    

    X = df_aderentes[features].fillna("nao_informado")
    y = df_aderentes[target]

    # 3. Pré-processamento e Modelo
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Treinar Modelo
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model.fit(X_train, y_train)

    # 5. Prever candidatos não contratados (aderente == 0)
    nao_aderentes = df_completo_exportado[df_completo_exportado["aderente"] == 0].copy()
    X_novos = nao_aderentes[features].fillna("nao_informado")
    probas_novos = model.predict_proba(X_novos)[:, 1]
    nao_aderentes["similaridade"] = probas_novos

    # 6. Top 5 Candidatos com maior aderência
    top_5 = nao_aderentes.sort_values(by="similaridade", ascending=False).head(5)
    media_sim = round(top_5["similaridade"].mean() * 100, 1)

    # 7. Exibir resultado com estilo
    st.markdown(f"""
    <div style='text-align:center; margin-top: 20px; margin-bottom: 10px;'>
        <h3 style='color:#0a0d2b;'>💡 Top 5 Candidatos NÃO Contratados com Maior Aderência</h3>
        <p style='font-size: 28px; font-weight: bold; color: #2c3e50;'>{media_sim}%</p>
    </div>
    """, unsafe_allow_html=True)

    # 8. Tabela de candidatos com estilo
    st.dataframe(
        top_5[[
            "nome_candidato", "titulo_profissional", "cliente", 
            "estado", "titulo_vaga", "similaridade"
        ]]
        .rename(columns={
            "nome_candidato": "Nome",
            "titulo_profissional": "Título Profissional",
            "cliente": "Cliente",
            "estado": "Estado",
            "titulo_vaga": "Vaga",
            "similaridade": "Aderência (%)"
        })
        .assign(**{
            "Aderência (%)": lambda df: (df["Aderência (%)"] * 100).round(1)
        })
        .style.format({"Aderência (%)": "{:.1f}%"})
    )

