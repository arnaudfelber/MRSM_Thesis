import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import combinations
import warnings
import time

warnings.filterwarnings("ignore", message="No frequency information was provided")

def calculate_rolling_window_regime_probabilities(y, X, window_size, k_regimes=2):
    #General_probabilities
    general_model = sm.tsa.MarkovRegression(y, k_regimes=k_regimes, exog=X, switching_variance=True)
    general_result = general_model.fit(maxiter=1000, disp=False)
    general_probabilities = general_result.smoothed_marginal_probabilities


    # Rolling probabilities
    rolling_probabilities_matrix = pd.DataFrame(np.nan, index=y.index, columns=[f'Regime {i}' for i in range(k_regimes)])

    # 3. Perform non-overlapping rolling window analysis
    for start in range(0, len(y), window_size):
        end = start + window_size
        if end > len(y):
            break # Stop if the window exceeds the data length
        y_window = y.iloc[start:end]
        X_window = X.iloc[start:end]

        try:
            msm_model = sm.tsa.MarkovRegression(y_window, k_regimes=k_regimes, exog=X_window, switching_variance=True)
            msm_result = msm_model.fit(maxiter=1000, disp=False)
            smoothed_probs = msm_result.smoothed_marginal_probabilities
            # Store probabilities in the matrix for the corresponding window
            rolling_probabilities_matrix.iloc[start:end, :] = smoothed_probs.values

        except Exception as e:
            print(f"Error fitting model for window {start}-{end}: {e}")

    rolling_probabilities_matrix.dropna(inplace=True)
    return {'General_Probabilities': general_probabilities,'Rolling_Probabilities': rolling_probabilities_matrix}

def calculate_expanding_window_regime_probabilities(y, X, k_regimes, increment_size, initial_data_size):
    # General Probabilities
    general_model = sm.tsa.MarkovRegression(y, k_regimes=k_regimes, exog=X, switching_variance=True)
    general_result = general_model.fit(maxiter=1000, disp=False)
    general_probabilities = general_result.smoothed_marginal_probabilities
    general_probabilities.columns = [f'Regime {i}' for i in range(k_regimes)]
    general_probabilities.iloc[initial_data_size:]
    # Expanding window probabilities
    expanding_probabilities_matrix = pd.DataFrame(np.nan, index=y.index, columns=[f'Regime {i}' for i in range(k_regimes)])

    # Perform expanding window analysis
    for end in range(initial_data_size, len(y) + 1, increment_size):
        start_time = time.time() # Record the start time of the iteration

        # Define the expanding window (start from initial_data_size)
        y_window = y.iloc[:end]
        X_window = X.iloc[:end]
        try:
            # Fit the model on the current window
            msm_model = sm.tsa.MarkovRegression(y_window, k_regimes=k_regimes, exog=X_window, switching_variance=True)
            msm_result = msm_model.fit(maxiter=100, disp=False)
            smoothed_probs = msm_result.smoothed_marginal_probabilities
            # Store probabilities for the last `increment_size` observations
            expanding_probabilities_matrix.iloc[end - increment_size:end, :] = smoothed_probs.iloc[-increment_size:].values
        except Exception as e:
            print(f"Error fitting model for window ending at {end}: {e}")
        # Log iteration duration
        end_time = time.time()
        iteration_time = end_time - start_time
        print(f"Iteration for window ending at {end} took {iteration_time:.4f} seconds")
    # Handle remaining observations if not divisible by increment_size
    if len(y) % increment_size != 0:
        y_window = y
        X_window = X
        try:
            msm_model = sm.tsa.MarkovRegression(y_window, k_regimes=k_regimes, exog=X_window, switching_variance=True)
            msm_result = msm_model.fit(maxiter=1000, disp=False)
            smoothed_probs = msm_result.smoothed_marginal_probabilities
            expanding_probabilities_matrix.iloc[-(len(y) % increment_size):, :] = smoothed_probs.iloc[-(len(y) % increment_size):].values
        except Exception as e:
            print(f"Error fitting final model: {e}")

    return {'General_Probabilities': general_probabilities,'Expanding_Window_Probabilities': expanding_probabilities_matrix}


def best_AIC_algorithm(y, X, lasso_selected_features):
    """
    Function to identify the best feature combination using AIC with a step-forward approach.
    """
    target = y['MOM_log_return']
    target.index = pd.date_range(start=target.index[0], periods=len(target), freq='W') # Weekly frequency
    
    all_features = X.columns[lasso_selected_features].tolist()
    best_aic = np.inf
    best_features = []
    remaining_features = set(all_features)
    aic_results = {}
    
    while remaining_features:
        current_best_aic = np.inf
        current_best_feature = None
        
        for feature in remaining_features:
            subset = best_features + [feature]
            X_subset = X[subset]
            
            try:
                msm_model = sm.tsa.MarkovRegression(target, k_regimes=2, exog=X_subset, switching_variance=True)
                msm_result = msm_model.fit(disp=False)
                current_aic = msm_result.aic
                aic_results[tuple(subset)] = current_aic
                
                if current_aic < current_best_aic:
                    current_best_aic = current_aic
                    current_best_feature = feature
            except Exception as e:
                print(f"Error with features {subset}: {e}")
        
        if current_best_aic < best_aic:
            best_aic = current_best_aic
            best_features.append(current_best_feature)
            remaining_features.remove(current_best_feature)
        else:
            break
    
    print("\n--- AIC Optimization Results ---")
    print("Best Feature Combination:")
    print(", ".join(best_features))
    print(f"Lowest AIC: {best_aic:.4f}")
    print("\nAll AIC Results:")
    for subset, aic in sorted(aic_results.items(), key=lambda x: x[1]):
        print(f"Features: {', '.join(subset)} | AIC: {aic:.4f}")
    return {"Best Features": best_features, "Lowest AIC": best_aic}


def best_BIC_algorithm(y, X, lasso_selected_features):
    """
    Function to identify the best feature combination using BIC with a step-forward approach.
    """
    target = y['MOM_log_return']
    target.index = pd.date_range(start=target.index[0], periods=len(target), freq='W') # Weekly frequency
    
    all_features = X.columns[lasso_selected_features].tolist()
    best_bic = np.inf
    best_features = []
    remaining_features = set(all_features)
    bic_results = {}
    
    while remaining_features:
        current_best_bic = np.inf
        current_best_feature = None
        
        for feature in remaining_features:
            subset = best_features + [feature]
            X_subset = X[subset]
            
            try:
                msm_model = sm.tsa.MarkovRegression(target, k_regimes=2, exog=X_subset, switching_variance=True)
                msm_result = msm_model.fit(disp=False)
                current_bic = msm_result.bic
                bic_results[tuple(subset)] = current_bic
                
                if current_bic < current_best_bic:
                    current_best_bic = current_bic
                    current_best_feature = feature
            except Exception as e:
                print(f"Error with features {subset}: {e}")
        
        if current_best_bic < best_bic:
            best_bic = current_best_bic
            best_features.append(current_best_feature)
            remaining_features.remove(current_best_feature)
        else:
            break
    
    print("\n--- BIC Optimization Results ---")
    print("Best Feature Combination:")
    print(", ".join(best_features))
    print(f"Lowest BIC: {best_bic:.4f}")
    print("\nAll BIC Results:")
    for subset, bic in sorted(bic_results.items(), key=lambda x: x[1]):
        print(f"Features: {', '.join(subset)} | BIC: {bic:.4f}")
    return {"Best Features": best_features, "Lowest BIC": best_bic}

def calculate_strategy_statistics_neutral_zone(data_w, regime_probabilities):
    """
    Calculate strategy statistics and create the regimes_binary DataFrame.

    Parameters:
        data_w (pd.DataFrame): Input data containing 'LOW_VOL', 'MOM', and 'MSCI' columns.
        regime_probabilities (pd.DataFrame): Regime probabilities with columns for each regime.

    Returns:
        dict: A dictionary containing statistics for each strategy and the selected regime.
        pd.DataFrame: The regimes_binary DataFrame.
    """

    # Calculate returns
    for col in ['LOW_VOL', 'MOM', 'MSCI']:
        data_w[f'{col}_Return'] = data_w[col].pct_change()
    data_w.dropna(inplace=True)

    # Annualized statistics
    stats = {}
    for col in ['LOW_VOL_Return', 'MOM_Return', 'MSCI_Return']:
        annualized_mean = data_w[col].mean() * 52
        annualized_variance = data_w[col].var() * 52
        annualized_std_dev = np.sqrt(annualized_variance)
        sharpe_ratio = annualized_mean / annualized_std_dev # Assume risk-free rate = 0
        stats[col] = {
            'Annualized Return': annualized_mean*100,
            'Annualized Variance': annualized_variance*100**2,
            'Annualized Std Dev': annualized_std_dev*100,
            'Sharpe Ratio': sharpe_ratio,
        }

    # Merge regime probabilities with data
    regime_probabilities.index = pd.to_datetime(regime_probabilities.index)
    data_w.index = pd.to_datetime(data_w.index)
    regime_probabilities.sort_index(inplace=True)
    data_w.sort_index(inplace=True)
    combined_df = pd.merge_asof(regime_probabilities, data_w, left_index=True, right_index=True)

    # Regime switching
    regime_switching_df = combined_df[[0, 1, 'LOW_VOL_Return', 'MOM_Return']]
    regime_switching_df.columns = ['Regime 0', 'Regime 1', 'LOW_VOL_Return', 'MOM_Return']

    regimes_binary = regime_switching_df.copy()
    
    regimes_binary['Regime 0'] = (regimes_binary['Regime 0'] > 0.9).astype(int)
    regimes_binary['Regime 1'] = (regimes_binary['Regime 1'] > 0.9).astype(int)
    for i in range(1, len(regimes_binary)): 
        # Check if the current row has both columns as 0
        if regimes_binary.iloc[i, regimes_binary.columns.get_loc('Regime 0')] == 0 and \
           regimes_binary.iloc[i, regimes_binary.columns.get_loc('Regime 1')] == 0:
            # Copy values from the previous row
            regimes_binary.iloc[i, regimes_binary.columns.get_loc('Regime 0')] = \
                regimes_binary.iloc[i - 1, regimes_binary.columns.get_loc('Regime 0')]
            regimes_binary.iloc[i, regimes_binary.columns.get_loc('Regime 1')] = \
                regimes_binary.iloc[i - 1, regimes_binary.columns.get_loc('Regime 1')]


    

    
    regimes_binary['LOW_VOL_Return_shifted'] = regimes_binary['LOW_VOL_Return'].shift(-1)
    regimes_binary['MOM_Return_shifted'] = regimes_binary['MOM_Return'].shift(-1)

    # Selected returns based on regimes
    selected_returns_df = regimes_binary['Regime 1'] * regimes_binary['LOW_VOL_Return_shifted'] + \
                          regimes_binary['Regime 0'] * regimes_binary['MOM_Return_shifted']
    dynamic_returns = selected_returns_df
    
    selected_returns = selected_returns_df.mean() * 52
    selected_var = selected_returns_df.var() * 52
    selected_std = np.sqrt(selected_var)
    selected_sharpe_ratio = selected_returns / selected_std # Assume risk-free rate = 0

    stats['Regime Switching Strategy'] = {
        'Annualized Return': selected_returns*100,
        'Annualized Variance': selected_var*100**2,
        'Annualized Std Dev': selected_std*100,
        'Sharpe Ratio': selected_sharpe_ratio,
    }

    return stats, regimes_binary, dynamic_returns
    
def neutral_zone_no_fill(data_w, regime_probabilities):
    """
    Calculate strategy statistics and create the regimes_binary DataFrame.

    Parameters:
        data_w (pd.DataFrame): Input data containing 'LOW_VOL', 'MOM', and 'MSCI' columns.
        regime_probabilities (pd.DataFrame): Regime probabilities with columns for each regime.

    Returns:
        dict: A dictionary containing statistics for each strategy and the selected regime.
        pd.DataFrame: The regimes_binary DataFrame.
    """

    # Calculate returns
    for col in ['LOW_VOL', 'MOM', 'MSCI']:
        data_w[f'{col}_Return'] = data_w[col].pct_change()
    data_w.dropna(inplace=True)

    # Annualized statistics
    stats = {}
    for col in ['LOW_VOL_Return', 'MOM_Return', 'MSCI_Return']:
        annualized_mean = data_w[col].mean() * 52
        annualized_variance = data_w[col].var() * 52
        annualized_std_dev = np.sqrt(annualized_variance)
        sharpe_ratio = annualized_mean / annualized_std_dev # Assume risk-free rate = 0
        stats[col] = {
            'Annualized Return': annualized_mean*100,
            'Annualized Variance': annualized_variance*100**2,
            'Annualized Std Dev': annualized_std_dev*100 ,
            'Sharpe Ratio': sharpe_ratio,
        }

    # Merge regime probabilities with data
    regime_probabilities.index = pd.to_datetime(regime_probabilities.index)
    data_w.index = pd.to_datetime(data_w.index)
    regime_probabilities.sort_index(inplace=True)
    data_w.sort_index(inplace=True)
    combined_df = pd.merge_asof(regime_probabilities, data_w, left_index=True, right_index=True)

    # Regime switching
    regime_switching_df = combined_df[[0, 1, 'LOW_VOL_Return', 'MOM_Return']]
    regime_switching_df.columns = ['Regime 0', 'Regime 1', 'LOW_VOL_Return', 'MOM_Return']

    regimes_binary = regime_switching_df.copy()
    
    regimes_binary['Regime 0'] = (regimes_binary['Regime 0'] > 0.9).astype(int)
    regimes_binary['Regime 1'] = (regimes_binary['Regime 1'] > 0.9).astype(int)


    

    
    regimes_binary['LOW_VOL_Return_shifted'] = regimes_binary['LOW_VOL_Return'].shift(-1)
    regimes_binary['MOM_Return_shifted'] = regimes_binary['MOM_Return'].shift(-1)

    # Selected returns based on regimes
    selected_returns_df = regimes_binary['Regime 1'] * regimes_binary['LOW_VOL_Return_shifted'] + \
                          regimes_binary['Regime 0'] * regimes_binary['MOM_Return_shifted']
    dynamic_returns = selected_returns_df
    
    selected_returns = selected_returns_df.mean() * 52
    selected_var = selected_returns_df.var() * 52
    selected_std = np.sqrt(selected_var)
    selected_sharpe_ratio = selected_returns / selected_std # Assume risk-free rate = 0

    stats['Regime Switching Strategy'] = {
        'Annualized Return': selected_returns*100,
        'Annualized Variance': selected_var*100**2,
        'Annualized Std Dev': selected_std*100,
        'Sharpe Ratio': selected_sharpe_ratio,
    }

    return stats, regimes_binary, dynamic_returns

def calculate_strategy_statistics(data_w, regime_probabilities):
    """
    Calculate strategy statistics and create the regimes_binary DataFrame.

    Parameters:
        data_w (pd.DataFrame): Input data containing 'LOW_VOL', 'MOM', and 'MSCI' columns.
        regime_probabilities (pd.DataFrame): Regime probabilities with columns for each regime.

    Returns:
        dict: A dictionary containing statistics for each strategy and the selected regime.
        pd.DataFrame: The regimes_binary DataFrame.
    """

    # Calculate returns
    for col in ['LOW_VOL', 'MOM', 'MSCI']:
        data_w[f'{col}_Return'] = data_w[col].pct_change()
    data_w.dropna(inplace=True)

    # Annualized statistics
    stats = {}
    for col in ['LOW_VOL_Return', 'MOM_Return', 'MSCI_Return']:
        annualized_mean = data_w[col].mean() * 52
        annualized_variance = data_w[col].var() * 52
        annualized_std_dev = np.sqrt(annualized_variance)
        sharpe_ratio = annualized_mean / annualized_std_dev # Assume risk-free rate = 0
        stats[col] = {
            'Annualized Mean (%)': annualized_mean*100 ,
            'Annualized Variance (%)': annualized_variance*100**2 , # variance is squared
            'Annualized Std Dev (%)': annualized_std_dev*100 ,
            'Sharpe Ratio': sharpe_ratio , # convert to percentage
        }

    # Merge regime probabilities with data
    regime_probabilities.index = pd.to_datetime(regime_probabilities.index)
    data_w.index = pd.to_datetime(data_w.index)
    regime_probabilities.sort_index(inplace=True)
    data_w.sort_index(inplace=True)
    combined_df = pd.merge_asof(regime_probabilities, data_w, left_index=True, right_index=True)

    # Regime switching
    regime_switching_df = combined_df[[0, 1, 'LOW_VOL_Return', 'MOM_Return']]
    regime_switching_df.columns = ['Regime 0', 'Regime 1', 'LOW_VOL_Return', 'MOM_Return']

    regimes_binary = regime_switching_df.copy()
    regimes_binary['Regime 0'] = (regimes_binary['Regime 0'] > regimes_binary['Regime 1']).astype(int)
    regimes_binary['Regime 1'] = (regimes_binary['Regime 1'] > regimes_binary['Regime 0']).astype(int)
    regimes_binary['LOW_VOL_Return_shifted'] = regimes_binary['LOW_VOL_Return'].shift(-1)
    regimes_binary['MOM_Return_shifted'] = regimes_binary['MOM_Return'].shift(-1)

    # Selected returns based on regimes
    selected_returns_df = regimes_binary['Regime 1'] * regimes_binary['LOW_VOL_Return'] + \
                          regimes_binary['Regime 0'] * regimes_binary['MOM_Return']
    dynamic_returns = selected_returns_df
    
    selected_returns = selected_returns_df.mean() * 52
    selected_var = selected_returns_df.var() * 52
    selected_std = np.sqrt(selected_var)
    selected_sharpe_ratio = selected_returns / selected_std # Assume risk-free rate = 0

    stats['Regime Switching Strategy'] = {
        'Annualized Mean (%)': selected_returns*100 ,
        'Annualized Variance (%)': selected_var*100**2 , # variance is squared
        'Annualized Std Dev (%)': selected_std*100 ,
        'Sharpe Ratio': selected_sharpe_ratio , # convert to percentage
    }

    return stats, regimes_binary, dynamic_returns

def analyze_regime_switches(regimes_binary):
    """
    Analyze regime switches in the given regimes_binary DataFrame.

    Parameters:
        regimes_binary (pd.DataFrame): DataFrame with binary regime indicators.

    Returns:
        tuple: A tuple containing the number of weeks allowed to MOM, LOW_VOL portfolios,
               the number of switches made, and a DataFrame of switch dates and corresponding regimes.
    """

    # Count the number of weeks in each regime
    N_time_MOM = np.sum(regimes_binary['Regime 0'])
    N_time_LOW_VOL = np.sum(regimes_binary['Regime 1'])

    # Identify regime switches
    Regime_0 = regimes_binary['Regime 0']
    switch_df = pd.DataFrame()
    switch_date = []
    switch_0 = []
    switch_count = 0
    for i in range(len(Regime_0) - 1):
        if Regime_0.iloc[i + 1] != Regime_0.iloc[i]:  # Access elements by position using .iloc
            switch_count += 1
            switch_date.append(regimes_binary.index[i + 1])  # Append the date after the switch
            switch_0.append(Regime_0.iloc[i + 1])  # Access element by position using .iloc

    # Create a DataFrame of switch dates and corresponding regimes
    if switch_date:
        switch_df = pd.DataFrame(list(zip(switch_date, switch_0)), columns=['Date', 'Switch'])

    # Get the start and end dates of the analysis period
    start_date = regimes_binary.index[0].strftime('%Y-%m-%d')
    end_date = regimes_binary.index[-1].strftime('%Y-%m-%d')

    return N_time_MOM, N_time_LOW_VOL, switch_count, start_date, end_date, switch_df

def plot_regime_activation(regimes_binary, title):
    """
    Plot the activation of regimes over time.

    Parameters:
        regimes_binary (pd.DataFrame): DataFrame with binary regime indicators.
    """

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(20,6))

    # Get the dates
    dates = regimes_binary.index

    # Convert dates to datetime objects for easier manipulation
    dates_dt = pd.to_datetime(dates)

    # Calculate the width of each bar
    width = (dates_dt[1] - dates_dt[0]).total_seconds() / (60 * 60 * 24)  # width in days

    # Get the regime activation values
    regime_0 = regimes_binary['Regime 0']
    regime_1 = regimes_binary['Regime 1']

    # Create arrays for the bar heights and colors
    heights = np.ones(len(dates))
    colors = ['blue' if r0 == 1 else 'red' for r0, r1 in zip(regime_0, regime_1)]

    # Create lists of indices for each regime
    regime_0_indices = [i for i, (r0, r1) in enumerate(zip(regime_0, regime_1)) if r0 == 1]
    regime_1_indices = [i for i, (r0, r1) in enumerate(zip(regime_0, regime_1)) if r1 == 1]

    # Create the bar plot
    ax.bar([dates_dt[i] for i in regime_0_indices], [heights[i] for i in regime_0_indices], width=width, color='salmon', label='Momentum Portfolio')
    ax.bar([dates_dt[i] for i in regime_1_indices], [heights[i] for i in regime_1_indices], width=width, color='skyblue', label='Low-volatility Portfolio')

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("")
    ax.set_yticks([])  # Remove y-axis ticks since we're only interested in colors

    # Add a legend
    ax.legend(loc='upper right', borderaxespad=0.)

    # Layout so plots do not overlap
    fig.tight_layout(rect=[0,0,0.8,1])  # Adjust layout to accommodate legend
    plt.savefig(f"images/{title}", dpi=300)
    # Show the plot
    plt.show()