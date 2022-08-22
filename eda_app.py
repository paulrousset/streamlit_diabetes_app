# Core Package
import streamlit as st

# Load EDA Packages
import pandas as pd

# Load Data Visualisation packages
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# Load Data
@st.cache
def load_data(data):
    df = pd.read_csv(data)
    return df


def run_eda_app():
    st.subheader("From Exploratory Data Analysis")
    df = load_data("data/diabetes_data_upload.csv")
    df_encoded = load_data("data/diabetes_data_upload_clean.csv")
    freq_df = load_data("data/freqdist_of_age_data.csv")

    submenu = st.sidebar.selectbox("Submenu", ["Descriptive", "Plots"])
    if submenu == "Descriptive":
        st.dataframe(df)

        with st.expander("Data Types"):
            st.dataframe(df.dtypes.astype(str))

        with st.expander("Descriptive Summary "):
            st.dataframe(df_encoded.describe())

        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Class Distribution"):
                st.dataframe(df['class'].value_counts())

        with col2:
            with st.expander("Gender Distribution"):
                st.dataframe(df['Gender'].value_counts())

    elif submenu == 'Plots':
        st.subheader("Plots")

        # Layouts
        col1, col2 = st.columns(2)

        with col1:
            with st.expander("Dist Plot of Gender"):
                # Using Seaborn
                fig = plt.figure()
                sns.countplot(df['Gender'])
                st.pyplot(fig)

                # p1 = px.pie(gen_df, names='Gender Type', values='Counts')
                # st.plotly_chart(p1, use_container_width=True)

        with col2:
            with st.expander("Gender Distribution"):
                gen_df = df['Gender'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ['Gender Type', 'Counts']
                gen_df['Percent'] = round(gen_df['Counts'] / gen_df['Counts'].sum(), 2)
                st.table(gen_df)

        # Layouts
        col3, col4 = st.columns(2)

        with col3:
            with st.expander("Dist Plot of Class"):
                # Using Seaborn
                fig = plt.figure()
                sns.countplot(df['class'])
                st.pyplot(fig)

                # pl = px.pie(gen_df, names='Gender Type', values='Counts')
                # st.plotly_chart(pl, use_container_width=True)

        with col4:
            with st.expander("class Distribution"):
                gen_df = df['class'].value_counts().to_frame()
                gen_df = gen_df.reset_index()
                gen_df.columns = ['class', 'Counts']
                gen_df['Percent'] = round(gen_df['Counts'] / gen_df['Counts'].sum(), 2)
                st.table(gen_df)

        # Freq Dist
        with st.expander("Frequency DIst of Age"):
            p1 = px.bar(freq_df, x='Age', y='count')
            st.plotly_chart(p1)

        # Outlier Detection
        with st.expander("Outlier Detection"):
            p2 = px.box(df, x='Age', color='Gender')
            st.plotly_chart(p2)

        # Correlation
        with st.expander("Correlation Plot"):
            corr_matrix = df_encoded.corr()
            # fig = plt.figure(figsize=(20, 10))
            # sns.heatmap(corr_matrix, annot=True)
            # st.pyplot(fig)

            p3 = px.imshow(corr_matrix)
            st.plotly_chart(p3)

    else:
        pass
