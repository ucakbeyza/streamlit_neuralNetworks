import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="YAĞIŞ TAHMİNİ PROJESİ", page_icon="🌧️", layout="wide")
st.markdown(
    """
    <style>
        .css-18e3th9 {
            background-color: #003366;
        }
        .css-1v3fvcr {
            background-color: #003366;
            color: white;
        }
        .css-1v3fvcr h1, .css-1v3fvcr h2, .css-1v3fvcr h3 {
            color: white;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
        }
        .stSidebar > .css-1jls4m5 {
            background-color: #003366;
        }
        .stFileUploader {
            color: #003366;
        }
    </style>
    """, unsafe_allow_html=True
)

def main():
    st.title("🌧️YAĞIŞ TAHMİNİ UYGULAMASI🌧️")

  
    st.sidebar.header("Veri Yükleme")
    uploaded_file = st.sidebar.file_uploader("CSV veya Excel Dosyası Yükleyin", type=["csv", "xlsx"])

    if uploaded_file:
        
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("Yüklenen Veri Seti:")
        st.write(df.head())

      
        st.subheader("Data Preview")
        st.write(df.head())

        st.subheader("Data Summary")
        st.write(df.describe())

        
        st.subheader("Filter Data")
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select column to filter by", columns)
        unique_values = df[selected_column].unique()
        selected_value = st.selectbox("Select value", unique_values)

        filtered_df = df[df[selected_column] == selected_value]
        st.write(filtered_df)

       
        st.subheader("Plot Data")
        x_column = st.selectbox("Select x-axis column", columns)
        y_column = st.selectbox("Select y-axis column", columns)

        if st.button("Generate Plot"):
            st.line_chart(filtered_df.set_index(x_column)[y_column])

     
        rainy_days = df[df['Rainfall'] > 0]
        if rainy_days.empty:
            st.error("Yağışlı günler bulunamadı. Lütfen farklı bir veri seti yükleyin.")
            return

        st.write("Yağışlı Günler:")
        st.write(rainy_days.head())

       
        X = rainy_days[['Temperature', 'Humidity', 'Wind Speed']]
        y = (rainy_days['Rainfall'] > rainy_days['Rainfall'].mean()).astype(int)  # Ortalama yağış miktarından fazla mı?

        # ölçekleme
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

      
        st.sidebar.header("Analiz Yöntemleri")
        method = st.sidebar.selectbox("Doğrulama Yöntemini Seçin", ["5-Fold Cross Validation", "%66-%34 Train/Test", "10-Fold Cross Validation"])

        if st.sidebar.button("Analizi Başlat"):
            if method == "5-Fold Cross Validation":
                cross_validate(X_scaled, y, n_splits=5)
            elif method == "%66-%34 Train/Test":
                train_test_split_analysis(X_scaled, y)
            elif method == "10-Fold Cross Validation":
                cross_validate(X_scaled, y, n_splits=10)

# Çapraz doğrulama fonksiyonu
def cross_validate(X, y, n_splits):
    st.write(f"{n_splits}-Fold Çapraz Doğrulama Sonuçları")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    param_grid = {
        "hidden_layer_sizes": [(10, 10), (20, 10)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.001, 0.01]
    }
    mlp = MLPClassifier(max_iter=200, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=kf, scoring="accuracy")
    grid_search.fit(X, y)

    st.write(f"En İyi Parametreler: {grid_search.best_params_}")
    st.write(f"En İyi Başarı Skoru: {grid_search.best_score_:.2f}")

    # Konfüzyon Matrisi
    y_pred = grid_search.best_estimator_.predict(X)
    cm = confusion_matrix(y, y_pred)
    st.write("Konfüzyon Matrisi:")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Az Yağış", "Çok Yağış"]).plot(ax=ax)
    st.pyplot(fig)

    # Yapay Sinir Ağı Görselleştirmesi
    draw_neural_network(grid_search.best_params_['hidden_layer_sizes'])

# Eğitim/test ayırma fonksiyonu
def train_test_split_analysis(X, y):
    st.write("%66-%34 Eğitim/Test Ayırma Sonuçları")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)

    param_grid = {
        "hidden_layer_sizes": [(10, 10), (20, 10)],
        "activation": ["relu"],
        "solver": ["adam"],
        "alpha": [0.001, 0.01]
    }
    mlp = MLPClassifier(max_iter=200, random_state=42)
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.best_estimator_.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f"Başarı Skoru: {(cm[0, 0] + cm[1, 1]) / cm.sum():.2f}")
    st.write("Konfüzyon Matrisi:")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["Az Yağış", "Çok Yağış"]).plot(ax=ax)
    st.pyplot(fig)


    draw_neural_network(grid_search.best_params_['hidden_layer_sizes'])


def draw_neural_network(hidden_layer_sizes):
    G = nx.DiGraph()

   
    layer_sizes = [3] + list(hidden_layer_sizes) + [1]  

    pos = {}
    node_id = 0  # Benzersiz düğüm ID'si
    for i, layer_size in enumerate(layer_sizes):
        for j in range(layer_size):
            node_name = f"L{i+1}_{j+1}"
            pos[node_id] = (i, j)
            G.add_node(node_id, label=node_name)
            node_id += 1

    
    for i in range(len(layer_sizes) - 1):
        start = sum(layer_sizes[:i])
        end = sum(layer_sizes[:i+1])
        next_start = end
        next_end = sum(layer_sizes[:i+2])
        for node1 in range(start, end):
            for node2 in range(next_start, next_end):
                G.add_edge(node1, node2)


    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold', arrows=True)
    plt.title("Yapay Sinir Ağı Modeli Görselleştirmesi")
    st.pyplot(plt)

if __name__ == "__main__":
    main()