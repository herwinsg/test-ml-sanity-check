import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.table import Table
#from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import wandb
import configparser
import argparse

def main(config_path):
    # --- Read Configuration ---
    config = configparser.ConfigParser()
    config.read(config_path)

    # --- Part 0: Setup and Imports ---
    print("--- Part 0: Initializing ---")
    wandb.login()
    #drive.mount('/content/drive')
    print("✅ Drive mounted.")

    # --- Part 1: Load Data and Engineer Features ---
    print("\n--- Part 1: Loading Data and Engineering Features ---")
    DATA_PATH = config.get('data', 'data_path')
    FULLINFO_FILE = os.path.join(DATA_PATH, config.get('data', 'fullinfo_file'))
    
    TARGET_COLS = ['logLbol_QSO', 'logBH_QSO', 'MASS_BEST_GAL', 'SFR_BEST_GAL']
    RANDOM_SEED = config.getint('model', 'random_seed')

    full_info_df = Table.read(FULLINFO_FILE).to_pandas()

    # Feature Engineering (as before)
    full_info_df['u_minus_r'] = full_info_df['u_sdss'] - full_info_df['r_sdss']
    full_info_df['r_minus_i'] = full_info_df['r_sdss'] - full_info_df['i_sdss']
    full_info_df['i_minus_z'] = full_info_df['i_sdss'] - full_info_df['z_sdss']
    full_info_df['z_minus_J'] = full_info_df['z_sdss'] - full_info_df['J_2mass']
    full_info_df['J_minus_H'] = full_info_df['J_2mass'] - full_info_df['H_2mass']
    full_info_df['H_minus_Ks'] = full_info_df['H_2mass'] - full_info_df['Ks_2mass']
    full_info_df['Ks_minus_W1'] = full_info_df['Ks_2mass'] - full_info_df['IRAC1']
    full_info_df['W1_minus_W2'] = full_info_df['IRAC1'] - full_info_df['IRAC2']

    RICH_FEATURE_COLS = [
        'redshift', 'u_sdss', 'r_sdss', 'i_sdss', 'z_sdss', 'J_2mass',
        'H_2mass', 'Ks_2mass', 'IRAC1', 'IRAC2', 'u_minus_r', 'r_minus_i',
        'i_minus_z', 'z_minus_J', 'J_minus_H', 'H_minus_Ks', 'Ks_minus_W1',
        'W1_minus_W2'
    ]
    
    final_df = full_info_df[RICH_FEATURE_COLS + TARGET_COLS].dropna()
    X = final_df[RICH_FEATURE_COLS].values
    Y = final_df[TARGET_COLS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=config.getfloat('model', 'test_size'), random_state=RANDOM_SEED
    )
    
    x_scaler = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_test_s = x_scaler.transform(X_test)
    print("✅ Data preprocessing complete.")

    # --- Part 2: Run Sanity Check with Random Forest ---
    print("\n--- Part 2: Running Sanity Check ---")
    
    wandb.init(project=config.get('wandb', 'project_name'), config=dict(config.items('model')))
    
    n_targets = y_train.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(n_targets * 7, 6))
    if n_targets == 1: axes = [axes]

    for i, target_name in enumerate(TARGET_COLS):
        print(f"\n--- Training for Target: {target_name} ---")
        ax = axes[i]
        
        model = RandomForestRegressor(
            n_estimators=config.getint('model', 'n_estimators'),
            max_depth=config.getint('model', 'max_depth'),
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        model.fit(X_train_s, y_train[:, i])
        
        preds = model.predict(X_test_s)
        r2 = r2_score(y_test[:, i], preds)
        print(f"  -> R-squared Score for {target_name}: {r2:.4f}")
        wandb.log({f"{target_name}_r2_score": r2})

        # Plotting
        min_val = min(y_test[:, i].min(), preds.min())
        max_val = max(y_test[:, i].max(), preds.max())
        plot_range = [min_val - (max_val - min_val)*0.05, max_val + (max_val - min_val)*0.05]
        sns.scatterplot(x=y_test[:, i], y=preds, ax=ax, alpha=0.3, s=15, edgecolor='none')
        ax.plot(plot_range, plot_range, 'r--', lw=2, label="1:1 Line")
        ax.set_xlim(plot_range); ax.set_ylim(plot_range)
        ax.set_xlabel(f"True {target_name}"); ax.set_ylabel(f"Predicted {target_name}")
        ax.set_title(f"Random Forest ({target_name})\nR$^2$ = {r2:.3f}")
        ax.legend(); ax.grid(True, linestyle='--', alpha=0.6)

    wandb.log({"predictions_plot": plt})
    plt.tight_layout()
    plt.show()
    
    wandb.finish()
    print("\n✅ Final sanity check complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a sanity check with a Random Forest Regressor.")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    main(args.config)