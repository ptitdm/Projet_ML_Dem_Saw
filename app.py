import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# ── Configuration de la page ───────────────────────────────────────────────────
st.set_page_config(
    page_title="❤️ Heart Disease ML - IFOAD",
    page_icon="https://img.icons8.com/color/48/heart-with-pulse.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={}
)

# ── CSS personnalisé ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ══════════════════════════════════════════
       CACHER UNIQUEMENT LE BOUTON DEPLOY
       (on garde le reste du header intact)
    ══════════════════════════════════════════ */
    .stDeployButton,
    [data-testid="stDeployButton"],
    button[data-testid="baseButton-header"][title="Deploy"],
    button[title="Deploy this app"] { display: none !important; }

    /* ══════════════════════════════════════════
       IMAGE DE FOND
    ══════════════════════════════════════════ */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1579154204601-01588f351e67?w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }

    /* ══════════════════════════════════════════
       OVERLAY SEMI-TRANSPARENT UNIVERSEL
    ══════════════════════════════════════════ */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background: rgba(240, 244, 255, 0.55);
        backdrop-filter: blur(1px);
        -webkit-backdrop-filter: blur(1px);
        z-index: 0;
        pointer-events: none;
    }

    /* Contenu toujours au-dessus de l'overlay */
    .block-container {
        position: relative;
        z-index: 1;
        padding-top: 2rem;
    }

    /* ══════════════════════════════════════════
       TEXTES — noir profond, lisible sur fond clair
    ══════════════════════════════════════════ */
    .stMarkdown, .stMarkdown p,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
    .stMarkdown li, .stMarkdown strong,
    p, span, div, label,
    .stSelectbox label, .stSlider label,
    .stRadio label, .stCheckbox label,
    [data-testid="stText"], [data-testid="stMarkdownContainer"] {
        color: #0d0d1a !important;
    }
    .stSelectbox div, .stSlider div,
    .stNumberInput div, .stTextInput div {
        color: #0d0d1a !important;
    }

    /* ══════════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.97) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-right: 1px solid rgba(0,0,0,0.09);
        box-shadow: 3px 0 18px rgba(0,0,0,0.12);
    }
    [data-testid="stSidebar"] * {
        color: #0d0d1a !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        font-weight: 600;
    }

    /* ══════════════════════════════════════════
       METRIQUES
    ══════════════════════════════════════════ */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.97) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 12px;
        border: 1px solid rgba(0,0,0,0.09);
        border-left: 4px solid #e74c3c;
        box-shadow: 0 2px 14px rgba(0,0,0,0.09);
    }
    div[data-testid="metric-container"] * {
        color: #0d0d1a !important;
    }

    /* ══════════════════════════════════════════
       ONGLETS INTERNES (tabs dans les pages)
    ══════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(6px);
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 700;
        color: #0d0d1a !important;
        border: 1px solid rgba(0,0,0,0.11);
        box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    }
    .stTabs [aria-selected="true"] {
        background: #c0392b !important;
        color: #ffffff !important;
        border-color: #c0392b !important;
    }

    /* ══════════════════════════════════════════
       TABLEAUX
    ══════════════════════════════════════════ */
    .stDataFrame, [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.97) !important;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }

    /* ══════════════════════════════════════════
       ALERTES
    ══════════════════════════════════════════ */
    .stAlert {
        background: rgba(255, 255, 255, 0.97) !important;
        backdrop-filter: blur(8px);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    }
    .stAlert, .stAlert * { color: #0d0d1a !important; }

    /* ══════════════════════════════════════════
       BOUTONS
    ══════════════════════════════════════════ */
    .stButton > button {
        background: #c0392b !important;
        color: #ffffff !important;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        box-shadow: 0 3px 12px rgba(192,57,43,0.35);
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #a93226 !important;
        box-shadow: 0 5px 18px rgba(192,57,43,0.45);
        transform: translateY(-1px);
    }

    /* ══════════════════════════════════════════
       CLASSES METIER
    ══════════════════════════════════════════ */
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #c0392b !important;
        text-align: center;
        padding: 10px 0;
        text-shadow: 0 1px 6px rgba(0,0,0,0.15);
    }
    .subtitle {
        font-size: 1rem;
        color: #444455 !important;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: rgba(255,255,255,0.97);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        margin: 5px;
        border: 1px solid rgba(0,0,0,0.09);
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #c0392b !important; }
    .metric-label { font-size: 0.85rem; color: #333 !important; }

    .prediction-box-disease {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: #ffffff !important;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(192,57,43,0.40);
    }
    .prediction-box-disease * { color: #ffffff !important; }

    .prediction-box-healthy {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: #ffffff !important;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(39,174,96,0.40);
    }
    .prediction-box-healthy * { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Chargement des données ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('../data/heart_disease.csv')
    except:
        df = pd.read_csv('data/heart_disease.csv')
    return df

@st.cache_data
def train_all_models(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
    }

    results = []
    trained = {}
    roc_data = {}

    for name, model in models.items():
        if name in ['Decision Tree', 'Random Forest', 'AdaBoost']:
            X_tr, X_te = X_train, X_test
        else:
            X_tr, X_te = X_train_sc, X_test_sc
        model.fit(X_tr, y_train)
        trained[name] = model
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)[:, 1]
        results.append({
            'Modèle': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Précision': precision_score(y_test, y_pred),
            'Rappel': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_prob)
        })
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_prob), y_pred, y_prob)

    results_df = pd.DataFrame(results).set_index('Modèle')
    return trained, scaler, results_df, roc_data, X_test, y_test

df = load_data()
trained_models, scaler, results_df, roc_data, X_test, y_test = train_all_models(df)
feature_names = [c for c in df.columns if c != 'target']

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    # ✅ ICÔNE REMPLACÉE : cardiogramme ECG au lieu du cœur simple
    st.image("https://img.icons8.com/color/200/heart-with-pulse.png", width=100)
    st.markdown("## ❤️ Heart Disease ML")
    st.markdown("**Projet_Dem-Sawadogo_ML**")
    st.divider()
    st.markdown("### Navigation")
    page = st.radio("", [
        "🏠 Accueil",
        "📊 Analyse Exploratoire",
        "🤖 Comparaison des Modèles",
        "🔮 Prédiction Patient"
    ])
    st.divider()
    st.markdown(f"📦 **Dataset:** {df.shape[0]} patients, {df.shape[1]-1} features")
    st.markdown(f"✅ **Maladie:** {df['target'].sum()} ({df['target'].mean():.1%})")
    st.markdown(f"❌ **Sains:** {(df['target']==0).sum()} ({1-df['target'].mean():.1%})")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    st.markdown('<div class="main-title">❤️ Projet ML — Maladies Cardiaques</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Machine Learning de prediction</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Patients", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1]-1)
    with col3:
        best_auc = results_df['AUC-ROC'].max()
        st.metric("Meilleur AUC", f"{best_auc:.3f}")
    with col4:
        best_acc = results_df['Accuracy'].max()
        st.metric("Meilleure Acc.", f"{best_acc:.3f}")

    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("### Les Algorithmes De Comparaisons")
        st.info("""
        En utilisant le dataset **Heart Disease UCI**:

        Voici les algorithmes **6 algorithmes** de machine learning qui sont comparés :
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Decision Tree
        - Random Forest
        - AdaBoost
        """)

    with col_b:
        st.markdown("### 📋 Description des Features")
        feature_desc = {
            'age': 'Âge en années', 'sex': 'Sexe (1=H, 0=F)',
            'cp': 'Type douleur thoracique (0-3)',
            'trestbps': 'Pression artérielle (mmHg)',
            'chol': 'Cholestérol (mg/dl)',
            'fbs': 'Glycémie à jeun >120 (0/1)',
            'restecg': 'ECG au repos (0-2)',
            'thalach': 'Fréq. cardiaque max',
            'exang': 'Angine exercice (0/1)',
            'oldpeak': 'Dépression segment ST',
            'slope': 'Pente segment ST (0-2)',
            'ca': 'Nb vaisseaux (0-3)',
            'thal': 'Thalassémie (1-3)',
            'target': '🎯 Maladie (1=oui, 0=non)'
        }
        desc_df = pd.DataFrame(list(feature_desc.items()), columns=['Feature', 'Description'])
        st.dataframe(desc_df, hide_index=True, height=380)

    st.divider()
    st.markdown("### 📊 Tableau Récapitulatif des Résultats")
    styled = results_df.style.background_gradient(cmap='YlOrRd', vmin=0.5, vmax=1.0).format("{:.4f}")
    st.dataframe(styled, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE EXPLORATOIRE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse Exploratoire":
    st.markdown("## 📊 Analyse Exploratoire des Données")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔢 Vue d'ensemble",
        "👤 Démographie",
        "Variables cliniques",
        "🔗 Corrélations"
    ])

    with tab1:
        st.markdown("### Statistiques descriptives")
        st.dataframe(df.describe().round(2), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Distribution de la cible")
            fig, ax = plt.subplots(figsize=(5, 4))
            counts = df['target'].value_counts()
            colors = ['#2ecc71', '#e74c3c']
            wedges, texts, autotexts = ax.pie(
                counts, labels=['Pas de maladie', 'Maladie'],
                autopct='%1.1f%%', colors=colors, startangle=90,
                wedgeprops=dict(edgecolor='white', linewidth=2)
            )
            ax.set_title('Répartition maladie cardiaque', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Distribution de l'âge")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(df[df['target']==0]['age'], bins=15, alpha=0.7, color='#2ecc71', label='Pas maladie', edgecolor='white')
            ax.hist(df[df['target']==1]['age'], bins=15, alpha=0.7, color='#e74c3c', label='Maladie', edgecolor='white')
            ax.set_xlabel('Âge'); ax.set_ylabel('Fréquence')
            ax.set_title('Distribution de l\'âge par groupe', fontweight='bold')
            ax.legend()
            st.pyplot(fig)
            plt.close()

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Maladie selon le sexe")
            fig, ax = plt.subplots(figsize=(6, 4))
            sex_target = df.groupby('sex')['target'].value_counts(normalize=True).unstack()
            sex_target.index = ['Femme', 'Homme']
            sex_target.columns = ['Pas maladie', 'Maladie']
            sex_target.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], edgecolor='white')
            ax.set_xticklabels(['Femme', 'Homme'], rotation=0)
            ax.set_ylabel('Proportion'); ax.set_title('Taux de maladie par sexe', fontweight='bold')
            ax.legend(); ax.set_ylim(0, 1)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("#### Type douleur thoracique vs Maladie")
            fig, ax = plt.subplots(figsize=(6, 4))
            cp_rate = df.groupby('cp')['target'].mean() * 100
            cp_labels = {0: 'Asympt.', 1: 'Atypique', 2: 'Typique', 3: 'Non-card.'}
            ax.bar([cp_labels.get(i, str(i)) for i in cp_rate.index],
                   cp_rate.values, color=['#3498db','#9b59b6','#e67e22','#e74c3c'], edgecolor='white')
            ax.set_ylabel('Taux de maladie (%)'); ax.set_title('Douleur thoracique vs Maladie', fontweight='bold')
            for i, v in enumerate(cp_rate.values):
                ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Angine à l'exercice vs Maladie")
            fig, ax = plt.subplots(figsize=(5, 4))
            exang_rate = df.groupby('exang')['target'].mean() * 100
            bars = ax.bar(['Non (0)', 'Oui (1)'], exang_rate.values,
                          color=['#2ecc71', '#e74c3c'], edgecolor='white', width=0.5)
            for bar, v in zip(bars, exang_rate.values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                        f'{v:.1f}%', ha='center', fontweight='bold')
            ax.set_ylabel('Taux de maladie (%)'); ax.set_title('Angine exercice vs Maladie', fontweight='bold')
            st.pyplot(fig)
            plt.close()

        with col4:
            st.markdown("#### Glycémie à jeun vs Maladie")
            fig, ax = plt.subplots(figsize=(5, 4))
            fbs_rate = df.groupby('fbs')['target'].mean() * 100
            bars = ax.bar(['≤120 mg/dl (0)', '>120 mg/dl (1)'], fbs_rate.values,
                          color=['#3498db', '#e67e22'], edgecolor='white', width=0.5)
            for bar, v in zip(bars, fbs_rate.values):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                        f'{v:.1f}%', ha='center', fontweight='bold')
            ax.set_ylabel('Taux de maladie (%)'); ax.set_title('Glycémie à jeun vs Maladie', fontweight='bold')
            st.pyplot(fig)
            plt.close()

    with tab3:
        st.markdown("#### Valeurs cliniques — Maladie vs Pas de maladie")
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        for ax, (var, title) in zip(axes, [
            ('trestbps', 'Pression artérielle\n(mmHg)'),
            ('chol', 'Cholestérol\n(mg/dl)'),
            ('thalach', 'Fréq. cardiaque max\n(bpm)')
        ]):
            data_groups = [df[df['target']==0][var], df[df['target']==1][var]]
            bp = ax.boxplot(data_groups, patch_artist=True, labels=['Pas maladie', 'Maladie'])
            for patch, c in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
                patch.set_facecolor(c); patch.set_alpha(0.7)
            ax.set_title(title, fontweight='bold')
        plt.suptitle('Comparaison des variables cliniques', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Statistiques moyennes par groupe")
        comp = df.groupby('target')[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']].mean().round(2)
        comp.index = ['Pas maladie', 'Maladie']
        st.dataframe(comp, use_container_width=True)

    with tab4:
        st.markdown("#### Matrice de corrélation")
        fig, ax = plt.subplots(figsize=(12, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=ax, mask=mask, linewidths=0.5,
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Matrice de Corrélation', fontsize=13, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### Top corrélations avec la variable cible (target)")
        corr_target = df.corr()['target'].drop('target').abs().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#e74c3c' if df.corr()['target'][feat] > 0 else '#3498db' for feat in corr_target.index]
        ax.barh(corr_target.index, corr_target.values, color=colors, edgecolor='white')
        ax.set_xlabel('|Corrélation|'); ax.set_title('Corrélations avec la cible', fontweight='bold')
        ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — COMPARAISON DES MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Comparaison des Modèles":
    st.markdown("## 🤖 Comparaison des Algorithmes de Classification")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Métriques",
        "📈 Courbes ROC",
        "🗺️ Matrices Confusion",
        "🌲 Importance Features"
    ])

    with tab1:
        st.markdown("### Tableau comparatif des métriques")
        styled = results_df.style.background_gradient(cmap='YlOrRd', vmin=0.5, vmax=1.0).format("{:.4f}")
        st.dataframe(styled, use_container_width=True)

        st.divider()
        st.markdown("### 🏆 Meilleur modèle par métrique")
        cols = st.columns(5)
        for col, metric in zip(cols, ['Accuracy', 'Précision', 'Rappel', 'F1-Score', 'AUC-ROC']):
            best = results_df[metric].idxmax()
            val = results_df[metric].max()
            with col:
                st.metric(metric, f"{val:.4f}", best.replace(' ', '\n'))

        st.divider()
        st.markdown("### Visualisation comparative")
        metrics = ['Accuracy', 'Précision', 'Rappel', 'F1-Score', 'AUC-ROC']
        colors_list = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        x = np.arange(len(results_df))
        width = 0.15

        fig, ax = plt.subplots(figsize=(13, 5))
        for i, (metric, color) in enumerate(zip(metrics, colors_list)):
            ax.bar(x + i*width, results_df[metric], width, label=metric, color=color, alpha=0.85, edgecolor='white')
        ax.set_xticks(x + width*2)
        ax.set_xticklabels([n.replace(' ', '\n') for n in results_df.index], fontsize=9)
        ax.set_ylabel('Score'); ax.set_ylim(0.4, 1.05)
        ax.set_title('Comparaison des Métriques par Algorithme', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.markdown("### Courbes ROC — Tous les modèles")
        fig, ax = plt.subplots(figsize=(9, 7))
        colors_roc = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6','#1abc9c']
        for (name, (fpr, tpr, auc, _, __)), color in zip(roc_data.items(), colors_roc):
            ax.plot(fpr, tpr, color=color, lw=2.5, label=f'{name} (AUC={auc:.3f})')
        ax.plot([0,1],[0,1],'k--',lw=1.5,label='Aléatoire (AUC=0.500)')
        ax.fill_between([0,1],[0,1],[0,1], alpha=0.05, color='gray')
        ax.set_xlabel('Taux de Faux Positifs (1-Spécificité)', fontsize=11)
        ax.set_ylabel('Taux de Vrais Positifs (Sensibilité)', fontsize=11)
        ax.set_title('Courbes ROC', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.info("""
        **Comment lire une courbe ROC :**
        - Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle
        - AUC = 1.0 → modèle parfait | AUC = 0.5 → modèle aléatoire
        - L'AUC mesure la capacité du modèle à distinguer les deux classes
        """)

    with tab3:
        st.markdown("### Matrices de Confusion")
        selected_model = st.selectbox("Choisir un modèle", list(trained_models.keys()))
        model = trained_models[selected_model]
        X_te = X_test if selected_model in ['Decision Tree', 'Random Forest', 'AdaBoost'] else scaler.transform(X_test)
        y_pred = model.predict(X_te)
        cm = confusion_matrix(y_test, y_pred)

        col1, col2 = st.columns([1, 1])
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Pas maladie', 'Maladie'],
                        yticklabels=['Pas maladie', 'Maladie'],
                        linewidths=1, annot_kws={'size': 14})
            ax.set_xlabel('Prédit', fontsize=12); ax.set_ylabel('Réel', fontsize=12)
            ax.set_title(f'Matrice de Confusion\n{selected_model}', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            tn, fp, fn, tp = cm.ravel()
            st.markdown("#### Décomposition de la matrice")
            st.metric("✅ Vrais Positifs (TP)", tp, help="Malades correctement identifiés")
            st.metric("✅ Vrais Négatifs (TN)", tn, help="Sains correctement identifiés")
            st.metric("⚠️ Faux Positifs (FP)", fp, help="Sains classifiés comme malades")
            st.metric("❌ Faux Négatifs (FN)", fn, help="Malades classifiés comme sains")

            st.markdown("---")
            st.write(f"**Accuracy :** {accuracy_score(y_test, y_pred):.4f}")
            st.write(f"**Précision :** {precision_score(y_test, y_pred):.4f}")
            st.write(f"**Rappel :** {recall_score(y_test, y_pred):.4f}")
            st.write(f"**F1-Score :** {f1_score(y_test, y_pred):.4f}")

    with tab4:
        st.markdown("### Importance des Features — Random Forest")
        rf = trained_models['Random Forest']
        fi = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(9, 6))
        colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi)))
        bars = ax.barh(fi['Feature'], fi['Importance'], color=colors_fi, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                    f'{bar.get_width():.3f}', va='center', fontsize=9)
        ax.set_xlabel('Importance (Gini)', fontsize=11)
        ax.set_title('Importance des Features - Random Forest', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        top3 = fi.tail(3)
        st.success(f"🌟 **Top 3 features :** {', '.join(top3['Feature'].tolist()[::-1])}")
# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prédiction Patient":
    st.markdown("## 🔮 Prédiction — Nouveau Patient")
    st.info("Renseignez les paramètres du patient pour obtenir une prédiction de risque cardiaque.")

    selected_model = st.selectbox(
        "🤖 Choisir l'algorithme",
        list(trained_models.keys()),
        index=list(trained_models.keys()).index('Support Vector Machine')
    )

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Informations générales**")
        age = st.slider("Âge (années)", 25, 80, 54)
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme (0)" if x == 0 else "Homme (1)")
        cp = st.selectbox("Type douleur thoracique (cp)",
                          [0, 1, 2, 3],
                          format_func=lambda x: {0:'0 - Asymptomatique',1:'1 - Angine atypique',
                                                  2:'2 - Angine typique',3:'3 - Non cardiaque'}[x])
        thal = st.selectbox("Thalassémie (thal)",
                            [1, 2, 3],
                            format_func=lambda x: {1:'1 - Normal',2:'2 - Défaut fixé',3:'3 - Défaut réversible'}[x])

    with col2:
        st.markdown("**🩸 Paramètres biologiques**")
        trestbps = st.slider("Pression artérielle repos (mmHg)", 90, 200, 130)
        chol = st.slider("Cholestérol (mg/dl)", 120, 570, 250)
        fbs = st.selectbox("Glycémie à jeun > 120 mg/dl", [0, 1], format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)")
        ca = st.selectbox("Nb vaisseaux colorés (ca)", [0, 1, 2, 3])

    with col3:
        st.markdown("**❤️ Paramètres cardiaques**")
        thalach = st.slider("Fréq. cardiaque max (bpm)", 70, 205, 150)
        restecg = st.selectbox("ECG au repos (restecg)", [0, 1, 2])
        exang = st.selectbox("Angine exercice (exang)", [0, 1], format_func=lambda x: "Non (0)" if x == 0 else "Oui (1)")
        oldpeak = st.slider("Dépression ST (oldpeak)", 0.0, 6.2, 1.0, 0.1)
        slope = st.selectbox("Pente ST (slope)", [0, 1, 2])

    st.divider()

    if st.button("🚀 Lancer la Prédiction", use_container_width=True, type="primary"):
        patient_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                   thalach, exang, oldpeak, slope, ca, thal]])

        model = trained_models[selected_model]
        if selected_model not in ['Decision Tree', 'Random Forest', 'AdaBoost']:
            patient_scaled = scaler.transform(patient_data)
        else:
            patient_scaled = patient_data

        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]

        st.divider()
        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box-disease">
                    ❤️‍🔥 RISQUE DE MALADIE CARDIAQUE DÉTECTÉ<br>
                    <span style="font-size:1rem; opacity:0.9">Probabilité : {probability[1]:.1%}</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-healthy">
                    💚 PAS DE MALADIE CARDIAQUE DÉTECTÉE<br>
                    <span style="font-size:1rem; opacity:0.9">Probabilité : {probability[0]:.1%}</span>
                </div>
                """, unsafe_allow_html=True)

        with col_res2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            proba_labels = ['Pas de maladie', 'Maladie']
            bar_colors = ['#2ecc71', '#e74c3c']
            bars = ax.barh(proba_labels, probability, color=bar_colors, edgecolor='white', height=0.5)
            for bar, prob in zip(bars, probability):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                        f'{prob:.1%}', va='center', fontweight='bold', fontsize=13)
            ax.set_xlim(0, 1.2); ax.set_xlabel('Probabilité')
            ax.set_title(f'Probabilités ({selected_model})', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.divider()
        st.markdown("### 📋 Récapitulatif du patient")
        patient_df = pd.DataFrame({
            'Feature': feature_names,
            'Valeur': patient_data[0]
        })
        st.dataframe(patient_df, hide_index=True, use_container_width=True)