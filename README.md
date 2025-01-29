Dynamic Portfolio Allocation using Markov Regime Switching Model (MRSM)

This repository contains the codebase for my Master Thesis, which explores the application of a Markov Regime Switching Model (MRSM) for dynamic portfolio allocation. The study focuses on switching between Minimum Volatility and Momentum strategies, optimizing risk-adjusted returns by dynamically adapting to market regimes.
Overview

The research aims to develop a quantitative investment strategy that leverages MRSM to enhance portfolio allocation by:

    - Identifying optimal regime switches between Minimum Volatility and Momentum strategies.
    - Applying advanced feature selection techniques like Elastic Net and Akaike Information Criterion (AIC) to optimize model inputs.
    - Implementing an Expanding Window approach to improve model calibration and reduce forward bias.
    - Evaluating performance metrics, including annualized return, Sharpe ratio, and volatility, in comparison to benchmark indices.
    - Evaluating relative switch performances, short term investment horizon and worst drawdown scenario

Repository Structure

This repository is structured into three main phases:
1. The data extraction through extraction.ipynb
2. The data preprocessing through data_preprocessing
3. The MRSM and Dynamic Portfolio Allocation strategy implementation and results through regime_switching

The code needs to be run in the same order, the initial dataset is the csv named data in the file data. With this data we can avoid the data extraction which is relied to a Bloomberg access. 


Key Findings

    - MRSM effectively captures market regimes, dynamically switching between Minimum Volatility and Momentum strategies.
    - Annualized return of 8.7% and a Sharpe ratio of 1.06, outperforming static benchmarks.
    - Low volatility (8.2%) compared to market indices, achieving better risk-adjusted returns.
    - Expanding Window approach reduces forward bias, improving model calibration over time.
    - Low transaction frequency, making the strategy practical for real-world investment applications.
    - The strategy has a goog long and short term return
    - The strategy is resilient in market downturn

Future Improvements

    - Testing additional explanatory variables (e.g., macroeconomic indicators, monetary policy signals).
    - Exploring alternative regime-switching models (e.g., Hidden Markov Models, neural networks).
    - Refining allocation strategy by incorporating a third regime for better adaptability.
    - Increasing dataset size to improve accuracy while balancing computational complexity.

Acknowledgments

Special thanks to Jean-Mark Lueder for insights on Minimum Volatility strategies and Bank Julius Bär & Co. Ltd. for supporting this financial research.

Special thanks to Pierre Collin Dufresne for the academic supervision and EPFL for the education.

Contact
If you have any questions or suggestions, feel free to reach out:

📧 arnaudfel@gmail.com
🔗 LinkedIn: www.linkedin.com/in/arnaud-felber-a20971300
📂 GitHub: https://github.com/arnaudfelber/MRSM_Thesis
