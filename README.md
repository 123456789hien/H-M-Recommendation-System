# H&M Fashion Recommendation System 👗

This repository contains the source code for a professional **H&M Fashion Recommendation App**, developed as part of a Master's Thesis. The application uses a **Three-Tower Neural Network** approach to provide intention-aware recommendations, achieving an AUC of **0.8201**.

## 🚀 Live Demo
The app is designed to be deployed on [[Streamlit Cloud](https://streamlit.io/cloud](https://share.streamlit.io/)).

## ✨ Key Features
- **Personalized Recommendations**: Tailored fashion suggestions based on unique user intention profiles.
- **Intention Discovery**: Explore the product catalog through 10 distinct fashion intentions (e.g., Casual, Formal, Trendy).
- **Fashion DNA Visualization**: Interactive radar charts showing the multi-dimensional style profile of each user.
- **E-commerce UI**: A clean, professional interface inspired by modern fashion retail applications.

## 🛠️ Technical Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Cosine Similarity for real-time ranking)
- **Visualization**: Plotly (Interactive Radar Charts)
- **Deployment**: Streamlit Cloud

## 📂 Project Structure
- `app.py`: The main Streamlit application code.
- `requirements.txt`: Python dependencies.
- `README.md`: Project documentation.

## ⚙️ How to Run Locally
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📊 Data Source
The data used in this project is based on the **H&M Personalized Fashion Recommendations** dataset from Kaggle, processed and enriched with intention-based metadata.

---
*Developed for Master's Thesis Research | 2026*
