# Telco-Churn-Intelligence
# 📡 Telco Customer Churn Intelligence Dashboard  A production-ready interactive dashboard designed to analyze, benchmark, and predict telecom customer churn in real time.   ### ⚡ Key Tech Stack: * **Frontend/UI:** Streamlit (Custom Premium Dark Theme) &amp; Plotly Analytics * **Machine Learning:** Scikit-Learn, XGBoost, LightGBM 
# 📡 Telco Customer Churn Intelligence Dashboard

A production-grade, interactive **Streamlit** enterprise dashboard designed to analyze, evaluate, and predict telecom customer churn in real time. Built with an optimized **scikit-learn**, **XGBoost**, and **LightGBM** machine learning backend, the application features an extensive exploratory data analysis engine, model benchmarking suites, and a live probability risk predictor wrapped in a fully customized **Premium Dark UI**.

---

## 🚀 Key Features

* **📊 Interactive EDA Engine (Tab 1):** Dynamically tracks 5 operational enterprise KPIs and visualizes structural churn drivers using optimized **Plotly Object** architectures (Donut breakdowns, overlapping demographic histograms, and multi-categorical pivot graphs).
* **⚖️ Enterprise Model Analytics Suite (Tab 2):** Benchmarks 5 different algorithms—*Logistic Regression, Random Forest, XGBoost, SVM (RBF), and LightGBM*. Includes interactive confusion matrices embedded with specialized dual-layered notations (`TN`, `FP`, `FN`, `TP`).
* **🎯 Live Churn Predictor & Risk Analysis (Tab 3):** Allows real-time inference via a 19-input profile compiler. Features automated feature alignment, dynamic encoding pipeline matchers, an analytical risk-factor checker, and a dynamic banner threshold system.
* **⚡ Production Optimization & Architecture:** * **Advanced Streamlit Caching:** Leverages `@st.cache_data` for heavy data manipulation layers and memory-safe `@st.cache_resource` for global machine learning models to prevent compute stalls.
    * **Automated Feature Alignment:** Guarantees structural matrix integrity during manual user profile inputs to avoid model breakage.
    * **Dynamic Sandbox Routing:** Features a multi-tiered file resolver (`_find_data_path()`) that scans target cloud paths and fallback local directories seamlessly.

---

## 🎨 Enterprise Design System

The application bypasses Streamlit's default components by injecting a highly responsive, custom global style layer via HTML injection to build a dark premium interface:
* **Blue (`#4C9BE8`):** Stabilized Accounts / Low Risk
* **Coral (`#E8634C`):** Churned Profiles / High-Critical Action Risk
* **Amber (`#F5C542`):** Moderate Intervention Boundaries
* **Cards (`#1A1D27`):** Elevated structural containers providing clear boundary contrast over a dark `#0F1117` backdrop.

---

## 🧠 Machine Learning Framework & Target Metrics

The data framework addresses severe **Class Imbalance (~2.8:1 ratio)** through proactive algorithmic optimizations:

1.  **Target Metric Priority (Recall ★):** In customer retention, a **False Negative** (failing to detect a churner) carries an exponentially higher customer lifetime value (CLV) replacement penalty than a **False Positive** (sending an unneeded promotion). The models prioritize high-sensitivity evaluation.
2.  **Imbalance Mitigation:** Configures intrinsic model weighting parameters (e.g., `class_weight='balanced'` for Random Forest, and calibrated `scale_pos_weight` ratios for XGBoost).
3.  **Strict Anti-Leakage Execution:** Enforces sequential constraints by calculating `StandardScaler` transformations uniquely over training splits before running scaling matrix mapping over testing splits.

### Algorithmic Benchmarks

| Model | Accuracy | Precision | Recall (Target ★) | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (RBF)** | 74.6% | 51.4% | **77.5%** | 61.8% |
| **XGBoost Classifier** | 75.9% | 53.3% | **75.4%** | 62.5% |
| **LightGBM Classifier** | 76.4% | 54.0% | **73.8%** | 62.4% |
| **Logistic Regression** | **80.5%** | **65.7%** | 55.9% | 60.4% |
| **Random Forest Classifier** | 77.8% | 60.3% | 47.6% | 53.2% |

*Note: While Logistic Regression yields the highest overall surface Accuracy, the **SVM** and **XGBoost** variants are vastly superior for production deployment due to their high recall capabilities in flag-catching churn behaviors.*

---

## 📁 Repository Structure

```text
├── app.py                                   # Core application script (Pipeline, Charts, Layouts)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Target Raw Dataset 
└── README.md                                # Repository documentation
