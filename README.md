# Projeto de Credit Score

Este projeto é o último do meu curso **Profissão Cientista de Dados** realizado na **EBAC** e tem como objetivo desenvolver um modelo de **Credit Score** utilizando **Regressão Logística** e outras técnicas de Machine Learning. O projeto foi estruturado em diferentes etapas, conforme detalhado abaixo.

## **Regressão Logística - Parte 1**
- Realizada a **Amostragem OOT (Out-of-Time)**.
- Análise **Descritiva Básica Univariada e Longitudinal**.
- Análise **Descritiva Bivariada**.
- **Desenvolvimento do Modelo**:
  - Tratamento de **missings** e **outliers**.
  - Identificação de **zeros estruturais**.
  - **Agrupamento de categorias**.
  - Análise de **Weight of Evidence (WOE)** e **Information Value (IV)**.
- **Ajuste do Modelo**:
  - Inicialmente, todas as variáveis foram incluídas na equação da **Regressão Logística**.
  - Em seguida, foram mantidas apenas as variáveis com **p-value menor que 5%** e **IV maior que 1%**.
- **Avaliação do Modelo** nas bases de treino e teste utilizando as métricas:
  - **AUC (Área Sob a Curva ROC)**
  - **GINI**
  - **KS (Kolmogorov-Smirnov)**
  - **Acurácia**

---

## **Regressão Logística - Parte 2**
- **Pré-processamento** semelhante ao descrito na Parte 1, mas utilizando **Sklearn Pipeline**.
- **Uso do PyCaret** para:
  - Pré-processar os dados.
  - Treinar um modelo baseado em **LightGBM**.
- **Avaliação do Modelo** criado com PyCaret e geração do **Pipeline** para futuras previsões.

---

## **Regressão Logística - Parte 2.1**
- Desenvolvimento de uma **função no Streamlit** para:
  - Carregar os dados.
  - Processar os dados utilizando o modelo **LightGBM** treinado anteriormente.
  - Disponibilizar um **botão para download** dos resultados.

---

## **Regressão Logística - Parte 2.2**
- **Aprimoramento do Pipeline** do notebook "Regressão Logística - Parte 2".
- Implementação de **redução de dimensionalidade** para otimização do modelo.
- Criação de uma **Regressão Logística** com o novo pré-processamento e **salvamento do modelo treinado**.

---

## **Regressão Logística - Parte 3**
- Implementação no **Streamlit** com os principais elementos dos notebooks anteriores:
  - **Análise Univariada e Bivariada**.
  - **Pipeline de Pré-processamento** (sem redução de dimensionalidade).
  - Modelo de **Regressão Logística Final** ("modelo_regressão_FINAL").
  - **Botão para download** dos resultados.

---

## **Regressão Logística - Parte 4**
- **Demonstração do funcionamento do Streamlit** por meio de um vídeo.
https://github.com/user-attachments/assets/3d9ae4f7-c018-4e32-a3c9-dcce69ccd327

---

## **Tecnologias Utilizadas**
- **Python** (Pandas, NumPy, Scikit-learn, PyCaret, LightGBM)
- **Streamlit** (Para interface interativa)
- **Jupyter Notebook** (Para experimentação e análise exploratória)

---

## **Conclusão**
Este projeto apresenta um fluxo completo para a construção de um modelo de **Credit Score**, desde a análise exploratória até a implantação via **Streamlit**. O uso do **PyCaret** e do **Sklearn Pipeline** otimiza o processamento, garantindo eficiência e reprodutibilidade. O modelo final pode ser utilizado para prever o risco de crédito com base nas características dos clientes.

