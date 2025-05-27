
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stroke Prediction Dashboard", layout="wide")

# Title
st.title("🧠 Stroke Prediction – Data Science Midterm Project")
st.markdown("A data-driven exploration of stroke risk factors based on health and demographic data.")

@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 75, 100],
                             labels=['<30', '30–45', '45–60', '60–75', '75+'])
    df['glucose_group'] = pd.cut(df['avg_glucose_level'],
                                 bins=[0, 100, 125, 150, 200, 300],
                                 labels=['<100', '100–125', '125–150', '150–200', '200+'])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filter the Data")
age_range = st.sidebar.slider("Age Range", 0, 100, (0, 100))
glucose_range = st.sidebar.slider("Glucose Level Range", 50, 300, (50, 300))
selected_smoking = st.sidebar.multiselect("Smoking Status", df['smoking_status'].dropna().unique(), default=df['smoking_status'].dropna().unique())

df_filtered = df[
    (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
    (df['avg_glucose_level'] >= glucose_range[0]) & (df['avg_glucose_level'] <= glucose_range[1]) &
    (df['smoking_status'].isin(selected_smoking))
]

# Dataset summary
st.subheader("📊 Dataset Summary (Filtered)")
st.write(df_filtered.describe())

# Visualizations
st.subheader("📈 Stroke Rate by Age Group")
stroke_by_age = df_filtered.groupby('age_group')['stroke'].mean().reset_index()
fig1, ax1 = plt.subplots()
sns.barplot(data=stroke_by_age, x='age_group', y='stroke', ax=ax1)
ax1.set_ylabel("Stroke Rate")
ax1.set_xlabel("Age Group")
st.pyplot(fig1)

st.subheader("📈 Stroke Rate by Glucose Level")
stroke_by_glucose = df_filtered.groupby('glucose_group')['stroke'].mean().reset_index()
fig2, ax2 = plt.subplots()
sns.barplot(data=stroke_by_glucose, x='glucose_group', y='stroke', ax=ax2)
ax2.set_ylabel("Stroke Rate")
ax2.set_xlabel("Glucose Group")
st.pyplot(fig2)

st.subheader("📈 Stroke Rate by Smoking Status")
stroke_by_smoking = df_filtered.groupby('smoking_status')['stroke'].mean().reset_index()
fig3, ax3 = plt.subplots()
sns.barplot(data=stroke_by_smoking, x='smoking_status', y='stroke', ax=ax3)
ax3.set_ylabel("Stroke Rate")
ax3.set_xlabel("Smoking Status")
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
st.pyplot(fig3)

st.subheader("📈 Stroke Rate by Health Conditions")
col1, col2 = st.columns(2)
with col1:
    stroke_by_hyper = df_filtered.groupby('hypertension')['stroke'].mean().reset_index()
    fig4, ax4 = plt.subplots()
    sns.barplot(data=stroke_by_hyper, x='hypertension', y='stroke', ax=ax4)
    ax4.set_title("Hypertension")
    ax4.set_ylabel("Stroke Rate")
    ax4.set_xlabel("Hypertension (0=No, 1=Yes)")
    st.pyplot(fig4)
with col2:
    stroke_by_heart = df_filtered.groupby('heart_disease')['stroke'].mean().reset_index()
    fig5, ax5 = plt.subplots()
    sns.barplot(data=stroke_by_heart, x='heart_disease', y='stroke', ax=ax5)
    ax5.set_title("Heart Disease")
    ax5.set_ylabel("Stroke Rate")
    ax5.set_xlabel("Heart Disease (0=No, 1=Yes)")
    st.pyplot(fig5)

# Stroke Risk Estimator
st.subheader("🧮 Personal Stroke Risk Estimator")

col1, col2, col3 = st.columns(3)
with col1:
    input_age = st.slider("Your Age", 0, 100, 50)
with col2:
    input_glucose = st.slider("Your Avg. Glucose Level", 50, 300, 120)
with col3:
    input_smoking = st.selectbox("Your Smoking Status", options=['never smoked', 'formerly smoked', 'smokes'])

col4, col5 = st.columns(2)
with col4:
    input_hypertension = st.radio("Do you have hypertension?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")
with col5:
    input_heart = st.radio("Do you have heart disease?", options=[0, 1], format_func=lambda x: "Yes" if x else "No")

# Filter for similar individuals
user_filter = df[
    (df['smoking_status'] == input_smoking) &
    (df['hypertension'] == input_hypertension) &
    (df['heart_disease'] == input_heart) &
    (df['age'].between(input_age - 10, input_age + 10)) &
    (df['avg_glucose_level'].between(input_glucose - 30, input_glucose + 30))
]

st.markdown("#### Matching individuals found in the dataset: **{}**".format(len(user_filter)))

if len(user_filter) >= 10:
    estimated_risk = user_filter['stroke'].mean()
    st.success(f"Estimated stroke rate among similar individuals: **{estimated_risk:.2%}**")
elif len(user_filter) > 0:
    estimated_risk = user_filter['stroke'].mean()
    st.warning("Limited data: Only {} similar individuals found.\nEstimated stroke rate: **{:.2%}**".format(len(user_filter), estimated_risk))
else:
    fallback_group = pd.cut([input_age], bins=[0, 30, 45, 60, 75, 100],
                            labels=['<30', '30–45', '45–60', '60–75', '75+'])[0]
    group_risk = stroke_by_age[stroke_by_age['age_group'] == fallback_group]['stroke'].values
    if len(group_risk) > 0:
        st.info(f"No exact matches found. Based on your age group ({fallback_group}), the average stroke rate is: **{group_risk[0]:.2%}**")
    else:
        st.error("No matching individuals found and no fallback age group available.")

# Insights and Summary
st.subheader("🧠 Key Insights")
st.markdown("""
1. **Stroke risk increases with age** – especially after 60 years.
2. **High glucose levels (>150 mg/dL)** are associated with a higher stroke rate.
3. **Hypertension and heart disease** both significantly increase stroke risk.
4. **Former smokers** have the highest stroke rate among smoking groups.
""")

st.subheader("📝 Summary")
st.markdown("""
These findings confirm known medical patterns and validate the dataset's reliability.  
Our visual analysis highlights the importance of monitoring age, glucose levels, and chronic conditions to assess stroke risk.
""")
