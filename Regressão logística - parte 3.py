import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from pycaret.classification import load_model, predict_model


def remove_outliers(df, columns, threshold=1.5):
    """Remove outliers das colunas numéricas contínuas com base no IQR."""
    for col in columns:
        if df[col].nunique() > 2:  # Aplica apenas a colunas não binárias
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Função para converter o df para excel
@st.cache_data
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.close()
    processed_data = output.getvalue()
    return processed_data


def main():
    st.title("Leitor de Arquivo Feather (.ftr)")
    
    uploaded_file = st.file_uploader("Carregue um arquivo .ftr", type=["ftr"])
    
    if uploaded_file is not None:
        df = pd.read_feather(uploaded_file)
        df['mau'] = df['mau'].astype(int)  # Garante que seja int
        st.write("### Visualização do DataFrame")
        st.dataframe(df)
        
        # Verificação inicial da coluna 'mau'
        st.write("Valores únicos em 'mau':", df['mau'].unique())
        
        # Seleção de variável para análise univariada
        numeric_columns = [col for col in df.columns if col not in ['data_ref', 'index']]
        selected_variable = st.selectbox("Selecione uma variável para análise univariada:", numeric_columns)
        
        if selected_variable:
            st.write(f"### Análise Univariada de {selected_variable}")
            st.bar_chart(df[selected_variable].value_counts())

        # Seleção de variável para análise bivariada com 'mau'
        if 'mau' in df.columns:
            bivariate_variable = st.selectbox("Selecione uma variável para análise bivariada com 'mau':", [col for col in df.columns if col not in ['data_ref', 'index', 'mau']])
            if bivariate_variable:
                st.write(f"### Análise Bivariada entre {bivariate_variable} e 'mau'")
                fig, ax = plt.subplots()
                
                if df[bivariate_variable].dtype == 'object':  # Se for categórica
                    sns.countplot(x=bivariate_variable, hue='mau', data=df, ax=ax)
                else:  # Se for numérica
                    sns.boxplot(x=df['mau'], y=df[bivariate_variable], ax=ax)
                
                st.pyplot(fig)

        # Anúncio de pré-processamento
        st.write('''A partir de agora iremos iniciar o pré-processamento dos dados,
        contendo os seguintes passos: Substituição de Nulos pela média, Identificação e 
        Remoção de Outliers com base no IQR, Transformador para Criação de Variáveis Dummies. A seguir, o resultado:''')
        
        # Identificar e substituir valores nulos
        df.fillna(df.mean(numeric_only=True), inplace=True)  # Para colunas numéricas
        df['mau'].fillna(0, inplace=True)  # Para a coluna 'mau'

        # Identificar e remover outliers (excluindo 'mau')
        numeric_columns = [col for col in df.columns if df[col].dtype != 'object' and col != 'mau']
        df = remove_outliers(df, numeric_columns)

        # Verificação após remoção de outliers
        st.write("Valores únicos em 'mau' após remoção de outliers:", df['mau'].unique())

        # Gerando as dummies
        df = pd.get_dummies(df, drop_first=True)  # drop_first para evitar multicolinearidade
        
        # Verificação após geração de dummies
        st.write("Colunas após geração de dummies:", df.columns)
        st.write("Valores únicos em 'mau' após geração de dummies:", df['mau'].unique())

        st.write("### DataFrame Após Pré-Processamento")
        st.dataframe(df)

        
        # Separando os conjuntos de treinamento e teste com base na OOT
        st.write("### Separando os Conjuntos de Treino e Teste")
        train = df[df['data_ref'] <= '2015-12-31']  # Até dezembro de 2015
        test = df[df['data_ref'] >= '2016-01-01']   # A partir de janeiro de 2016

        # Excluindo as colunas 'data_ref' e 'index'
        train = train.drop(columns=['data_ref', 'index'])
        test = test.drop(columns=['data_ref', 'index'])

        # Verificando os conjuntos
        st.write(f"Shape do conjunto de treino: {train.shape}")
        st.write(f"Shape do conjunto de teste: {test.shape}")

        # Carrega o modelo salvo
        model_saved = load_model('modelo_regressão_FINAL.pkl"')

        # Faz a predição
        predict = predict_model(model_saved, data=df_credit)

        # Converte para Excel e disponibiliza para download
        df_xlsx = to_excel(predict)
        st.download_button(label='📥 Download',
                           data=df_xlsx,
                           file_name='predict.xlsx')


if __name__ == "__main__":
    main()