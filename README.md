<!-- Able a quick return to the top page -->
<a name="readme-top"></a>

<!-- PROJECT LOGO -->
<br />


<h3 align="center">Dynamic Portfolio Allocation Using Markov Regime Switching Model: </h3>

  <p align="center">
     Momentum & Low-Volatility Strategies
    
  </p>
</div>

---

**Author:** [Arnaud Felber](https://people.epfl.ch/arnaud.felber?lang=en)


---

## 📖 Abstract


This research explores the application of dynamic portfolio allocation using the Markov Regime Switching Model (MRSM), focusing on transitions between Minimum Volatility and Momentum strategies. The goal of this dynamic strategy is to combine the complementarity stability of the Minimum Volatility strategy with the growth potential of the Momentum strategy. Thereby, the strategy balances risk and return by adapting to market conditions. The study employs advanced data preprocessing techniques to identify the variables subset. It includes feature selection through Elastic Net regression and Akaike Information Criterion (AIC) optimization. In addition, the Expanding Window method has been used to reduce potential forward bias and improve regime detection over time. The results underscore the efficacy of MRSM application to dynamic portfolio allocation strategy. The strategy achieves an annualized return of 8.7%, comparable to the MSCI World Index, and maintains a low volatility of 8.2%. With a Sharpe ratio of 1.06, the strategy outperforms other analysed strategies. This performance can be attributed to the model’s precision in regime allocation, its ability to capture long-term dependencies, and its performance gains relative to the switch decision. Additionally, the dynamic nature of the strategy mitigates drawdowns during periods of uncertainty and takes advantage of recovery and growth opportunities. Also, the low transaction frequency improves the practicality of the investment strategy by minimizing transactions and operational costs. Finally, this study contributes to the Markov Regime Switching Model applications and offers direction for future research in adaptive portfolio management.

---

## 💻 Getting Started  

To run this project, follow these steps:  

### **Requirements**  
Before running the code, ensure you have the following installed:  

- **Python** 3.8+  
- **Jupyter Notebook**  
- **Required Python libraries:**
numpy, pandas, matplotlib, scikit-learn, statsmodels

---

## 🚀 Running the Code  

Execute the Jupyter Notebooks **in the following order**:  

1️⃣ **Data Extraction** → `extraction.ipynb` *(Optional if using provided dataset)*  
2️⃣ **Data Preprocessing** → `data_preprocessing.ipynb`  
3️⃣ **MRSM Implementation & Results** → `regime_switching.ipynb`  

📌 **Note**: If you don't have Bloomberg API access, use the provided dataset (`data.csv`) inside the **data/** folder.  

---

## 📊 Key Findings

- **Dynamic Market Adaptation:** MRSM effectively captures market regimes, dynamically switching between Minimum Volatility and Momentum strategies.
- **Strong Performance:** Achieves an **annualized return of 8.7%** and a **Sharpe ratio of 1.06**.
- **Risk Management:** Maintains **low volatility (8.2%)**, delivering superior risk-adjusted returns compared to market indices.
- **Practical Implementation:** Low transaction frequency minimizes costs, making the strategy **feasible for real-world investment applications**.
- **Consistent Returns:** The strategy demonstrates **strong long-term and short-term performance**.
- **Resilience in Market Downturns:** Effectively mitigates drawdowns and recovers faster than static allocation models.

---
## 🔬 Future Improvements

- Testing additional explanatory variables (e.g., macroeconomic indicators, monetary policy signals).
- Exploring alternative regime-switching models (e.g., Hidden Markov Models, neural networks).
- Refining allocation strategy by incorporating a third regime for better adaptability.
- Increasing dataset size to improve accuracy while balancing computational complexity.



---
## 🙌 Acknowledgments

A special thanks to the following individuals and institutions for their contributions and support:

- **Jean-Mark Lueder** – Insights on Minimum Volatility strategies and support from Bank Julius Bär & Co. Ltd.
- **Pierre Collin-Dufresne** – Academic supervision and guidance.
- **EPFL** – Providing the educational foundation for this research.
