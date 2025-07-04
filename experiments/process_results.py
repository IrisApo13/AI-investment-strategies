#!/usr/bin/env python3
"""
Process backtest results from JSON files in the experiments directory.
Generates a table showing strategy scores by company.
"""

import json
import os
import glob
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_experiment_files(experiments_dir="."):
    """
    Process all JSON files in the experiments directory and extract strategy scores and excess returns.
    Processes all strategies and all iterations from each file.
    
    Returns:
        tuple: (results_scores, results_excess_returns) - Both dictionaries with structure {ticker: {strategy_name: value}}
    """
    results_scores = defaultdict(dict)
    results_excess_returns = defaultdict(dict)
    
    # Find all JSON files that match the backtest pattern
    json_files = glob.glob(os.path.join(experiments_dir, "backtest_*.json"))
    
    print(f"Found {len(json_files)} backtest files to process...")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract ticker and strategy information
            ticker = data.get('ticker')
            strategies = data.get('strategies', [])
            performance_history = data.get('performance_history', [])
            
            if not ticker or not strategies or not performance_history:
                print(f"Warning: Skipping {file_path} - missing required data")
                continue
            
            # Process all strategies and their corresponding performance entries
            for i, strategy in enumerate(strategies):
                strategy_name = strategy.get('name', f'Unknown Strategy {i+1}')
                
                # Get the corresponding performance entry (if available)
                if i < len(performance_history):
                    performance = performance_history[i]
                    score = performance.get('performance_score', 0)
                    excess_return = performance.get('excess_return', 0)
                else:
                    # If no corresponding performance entry, use the first one
                    performance = performance_history[0] if performance_history else {}
                    score = performance.get('performance_score', 0)
                    excess_return = performance.get('excess_return', 0)
                
                # Store the results
                results_scores[ticker][strategy_name] = score
                results_excess_returns[ticker][strategy_name] = excess_return
                
                print(f"Processed {ticker} - {strategy_name}: Score={score:.1f}, Excess Return={excess_return:.1f}%")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results_scores, results_excess_returns

def create_score_table(results):
    """
    Create a pandas DataFrame from the results for easy display.
    
    Args:
        results (dict): Dictionary with structure {ticker: {strategy_name: score}}
    
    Returns:
        pandas.DataFrame: Table with tickers as rows and strategies as columns
    """
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Sort by ticker
    df = df.sort_index()
    
    # Round scores to 1 decimal place for cleaner display
    df = df.round(2)
    
    return df

def plot_table(df):
    """
    Create a matplotlib table visualization of the results.
    Companies are columns, strategies are rows.
    
    Args:
        df (pandas.DataFrame): The results table
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Transpose the dataframe so companies become columns and strategies become rows
    df_transposed = df.T
    
    # Prepare table data
    table_data = []
    for strategy in df_transposed.index:
        row = [strategy[:30] + '...' if len(strategy) > 30 else strategy]  # Truncate long strategy names
        for col in df_transposed.columns:
            value = df_transposed.loc[strategy, col]
            if pd.isna(value):
                row.append("N/A")
            else:
                row.append(f"{value:.1f}")
        table_data.append(row)
    
    # Create column headers
    col_labels = ['Strategy'] + list(df_transposed.columns)
    
    # Create the table
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color code cells based on performance
    for i in range(len(table_data)):
        for j in range(1, len(col_labels)):  # Skip strategy name column
            cell = table[(i+1, j)]  # +1 because of header row
            try:
                value = float(table_data[i][j])
                # Check if value is NaN
                if pd.isna(value) or str(value).lower() == 'nan':
                    # Don't apply any color for NaN values
                    cell.set_facecolor('white')
                else:
                    # Color coding: Green for high scores, Red for low scores
                    if value >= 60:
                        cell.set_facecolor('#90EE90')  # Light green
                    elif value >= 40:
                        cell.set_facecolor('#FFE4B5')  # Light orange
                    elif value >= 20:
                        cell.set_facecolor('#FFB6C1')  # Light pink
                    else:
                        cell.set_facecolor('#FFCCCB')  # Light red
            except (ValueError, TypeError):
                # Handle NaN or other non-numeric values
                cell.set_facecolor('white')
    
    # Style header row
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor('#4CAF50')  # Green header
        cell.set_text_props(weight='bold', color='white')
    
    # Remove axis
    ax.axis('off')
    
    # Add title
    plt.title('Strategy Performance Scores by Company', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#90EE90', label='Excellent (≥60)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFE4B5', label='Good (40-59)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFB6C1', label='Fair (20-39)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFCCCB', label='Poor (<20)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('strategy_scores_table.png', dpi=300, bbox_inches='tight')
    print("Table visualization saved as 'strategy_scores_table.png'")
    
    # Show the plot
    plt.show()

def plot_heatmap(df):
    """
    Create a heatmap visualization of the strategy scores.
    
    Args:
        df (pandas.DataFrame): The results table
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create heatmap
    im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(df.columns)))
    ax.set_yticks(range(len(df.index)))
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.set_yticklabels(df.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score', rotation=270, labelpad=15)
    
    # Add text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            text = ax.text(j, i, f"{df.iloc[i, j]:.1f}",
                          ha="center", va="center", color="black", fontweight='bold')
    
    # Set title and labels
    ax.set_title('Strategy Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Strategy Type')
    ax.set_ylabel('Company')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('strategy_scores_heatmap.png', dpi=300, bbox_inches='tight')
    print("Heatmap visualization saved as 'strategy_scores_heatmap.png'")
    
    # Show the plot
    plt.show()

def print_table(df):
    """
    Print the results table in a formatted way with clear headers.
    Companies are columns, strategies are rows.
    
    Args:
        df (pandas.DataFrame): The results table
    """
    print("\n" + "="*100)
    print("STRATEGY PERFORMANCE SCORES BY COMPANY")
    print("="*100)
    
    if df.empty:
        print("No results found!")
        return
    
    # Transpose the dataframe so companies become columns and strategies become rows
    df_transposed = df.T
    
    # Get the actual company names from the transposed dataframe
    company_names = list(df_transposed.columns)
    
    # Create header row
    header = f"{'Strategy':<35} |"
    for company in company_names:
        header += f" {company:<8} |"
    print(header)
    print("-" * (len(header) + 10))
    
    # Print each row with formatted data
    for strategy in df_transposed.index:
        row = df_transposed.loc[strategy]
        
        # Start with strategy name (shortened for display)
        short_strategy = strategy[:34] if len(strategy) > 34 else strategy
        row_str = f"{short_strategy:<35} |"
        
        # Add each company score
        for company in company_names:
            score = row.get(company, 0)
            if pd.isna(score):
                row_str += f" {'N/A':<8} |"
            else:
                row_str += f" {score:.1f}{'':<7} |"
        
        print(row_str)
    
    print("-" * (len(header) + 10))
    
    # Print summary statistics
    print("\n" + "-"*100)
    print("SUMMARY STATISTICS")
    print("-"*100)
    
    # Average score by strategy
    print("\nAverage Score by Strategy:")
    strategy_means = df.mean().sort_values(ascending=False)
    for strategy, mean_score in strategy_means.items():
        print(f"  {strategy}: {mean_score:.1f}")
    
    # Average score by company
    print("\nAverage Score by Company:")
    company_means = df.mean(axis=1).sort_values(ascending=False)
    for company, mean_score in company_means.items():
        print(f"  {company}: {mean_score:.1f}")
    
    # Best performing strategy overall
    best_strategy = strategy_means.idxmax()
    best_score = strategy_means.max()
    print(f"\nBest Overall Strategy: {best_strategy} ({best_score:.1f})")
    
    # Best performing company overall
    best_company = company_means.idxmax()
    best_company_score = company_means.max()
    print(f"Best Overall Company: {best_company} ({best_company_score:.1f})")

def create_enhanced_csv(df):
    """
    Create an enhanced CSV with better formatting and headers.
    
    Args:
        df (pandas.DataFrame): The results table
    """
    # Create a copy for CSV export
    csv_df = df.copy()
    
    # Add a header row with strategy descriptions
    header_row = pd.DataFrame({
        'Momentum Strategy': ['Momentum-based strategy'],
        'Price Change Strategy': ['Price change-based strategy'],
        'MACD Strategy': ['MACD-based strategy'],
        'RSI Strategy': ['RSI-based strategy']
    }, index=['Description'])
    
    # Combine header with data
    enhanced_df = pd.concat([header_row, csv_df])
    
    return enhanced_df

def print_excess_returns_table(df_excess_returns):
    """
    Print the excess returns table in a formatted way with clear headers.
    Companies are columns, strategies are rows.
    
    Args:
        df_excess_returns (pandas.DataFrame): The excess returns table
    """
    print("\n" + "="*100)
    print("EXCESS RETURNS BY COMPANY AND STRATEGY")
    print("="*100)
    
    if df_excess_returns.empty:
        print("No excess returns data found!")
        return
    
    # Transpose the dataframe so companies become columns and strategies become rows
    df_transposed = df_excess_returns.T
    
    # Get the actual company names from the transposed dataframe
    company_names = list(df_transposed.columns)
    
    # Create header row
    header = f"{'Strategy':<35} |"
    for company in company_names:
        header += f" {company:<8} |"
    print(header)
    print("-" * (len(header) + 10))
    
    # Print each row with formatted data
    for strategy in df_transposed.index:
        row = df_transposed.loc[strategy]
        
        # Start with strategy name (shortened for display)
        short_strategy = strategy[:34] if len(strategy) > 34 else strategy
        row_str = f"{short_strategy:<35} |"
        
        # Add each company excess return
        for company in company_names:
            excess_return = row.get(company, 0)
            if pd.isna(excess_return):
                row_str += f" {'N/A':<8} |"
            else:
                row_str += f" {excess_return:+.1f}{'':<7} |"  # Use + sign for positive values
        
        print(row_str)
    
    print("-" * (len(header) + 10))
    
    # Print summary statistics for excess returns
    print("\n" + "-"*100)
    print("EXCESS RETURNS SUMMARY STATISTICS")
    print("-"*100)
    
    # Average excess return by strategy
    print("\nAverage Excess Return by Strategy:")
    strategy_means = df_excess_returns.mean().sort_values(ascending=False)
    for strategy, mean_excess in strategy_means.items():
        print(f"  {strategy}: {mean_excess:+.1f}%")
    
    # Average excess return by company
    print("\nAverage Excess Return by Company:")
    company_means = df_excess_returns.mean(axis=1).sort_values(ascending=False)
    for company, mean_excess in company_means.items():
        print(f"  {company}: {mean_excess:+.1f}%")
    
    # Best performing strategy overall (highest excess return)
    best_strategy = strategy_means.idxmax()
    best_excess = strategy_means.max()
    print(f"\nBest Overall Strategy (Excess Return): {best_strategy} ({best_excess:+.1f}%)")
    
    # Best performing company overall (highest excess return)
    best_company = company_means.idxmax()
    best_company_excess = company_means.max()
    print(f"Best Overall Company (Excess Return): {best_company} ({best_company_excess:+.1f}%)")
    
    # Count positive vs negative excess returns
    positive_count = (df_excess_returns > 0).sum().sum()
    total_count = df_excess_returns.size
    negative_count = total_count - positive_count
    
    print(f"\nExcess Return Distribution:")
    print(f"  Positive excess returns: {positive_count}/{total_count} ({positive_count/total_count*100:.1f}%)")
    print(f"  Negative excess returns: {negative_count}/{total_count} ({negative_count/total_count*100:.1f}%)")

def plot_excess_returns_table(df_excess_returns):
    """
    Create a matplotlib table visualization of the excess returns.
    Companies are columns, strategies are rows.
    
    Args:
        df_excess_returns (pandas.DataFrame): The excess returns table
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Transpose the dataframe so companies become columns and strategies become rows
    df_transposed = df_excess_returns.T
    
    # Prepare table data
    table_data = []
    for strategy in df_transposed.index:
        row = [strategy[:30] + '...' if len(strategy) > 30 else strategy]  # Truncate long strategy names
        for col in df_transposed.columns:
            value = df_transposed.loc[strategy, col]
            if pd.isna(value):
                row.append("N/A")
            else:
                row.append(f"{value:+.1f}")  # Use + sign for positive values
        table_data.append(row)
    
    # Create column headers
    col_labels = ['Strategy'] + list(df_transposed.columns)
    
    # Create the table
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Color code cells based on excess return performance
    for i in range(len(table_data)):
        for j in range(1, len(col_labels)):  # Skip strategy name column
            cell = table[(i+1, j)]  # +1 because of header row
            try:
                value = float(table_data[i][j])
                # Check if value is NaN
                if pd.isna(value) or str(value).lower() == 'nan':
                    # Don't apply any color for NaN values
                    cell.set_facecolor('white')
                else:
                    # Color coding: Green for positive, Red for negative
                    if value > 0:
                        cell.set_facecolor('#90EE90')  # Light green
                    elif value > -10:
                        cell.set_facecolor('#FFE4B5')  # Light orange
                    elif value > -20:
                        cell.set_facecolor('#FFB6C1')  # Light pink
                    else:
                        cell.set_facecolor('#FFCCCB')  # Light red
            except (ValueError, TypeError):
                # Handle NaN or other non-numeric values
                cell.set_facecolor('white')
    
    # Style header row
    for j in range(len(col_labels)):
        cell = table[(0, j)]
        cell.set_facecolor('#4CAF50')  # Green header
        cell.set_text_props(weight='bold', color='white')
    
    # Remove axis
    ax.axis('off')
    
    # Add title
    plt.title('Excess Returns by Company and Strategy', fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#90EE90', label='Positive (>0%)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFE4B5', label='Slight Negative (-10% to 0%)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFB6C1', label='Moderate Negative (-20% to -10%)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFCCCB', label='Large Negative (<-20%)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('excess_returns_table.png', dpi=300, bbox_inches='tight')
    print("Excess returns table visualization saved as 'excess_returns_table.png'")
    
    # Show the plot
    plt.show()

def main():
    """Main function to process results and display the table."""
    print("Processing backtest results from experiments directory...")
    
    # Process the files
    results_scores, results_excess_returns = process_experiment_files()
    
    if not results_scores or not results_excess_returns:
        print("No valid results found!")
        return
    
    # Create the performance scores table
    df_scores = create_score_table(results_scores)
    
    # Create the excess returns table
    df_excess_returns = create_score_table(results_excess_returns)
    
    # Display the performance scores results
    print_table(df_scores)
    
    # Display the excess returns results
    print_excess_returns_table(df_excess_returns)
    
    # Create matplotlib visualizations
    print("\nCreating matplotlib visualizations...")
    plot_table(df_scores)
    plot_excess_returns_table(df_excess_returns)
    #plot_heatmap(df_scores)
    
    # Save enhanced CSV with descriptions for scores
    enhanced_df_scores = create_enhanced_csv(df_scores)
    output_file_scores = "strategy_scores.csv"
    enhanced_df_scores.to_csv(output_file_scores)
    print(f"\nPerformance scores saved to {output_file_scores}")
    
    # Save enhanced CSV with descriptions for excess returns
    enhanced_df_excess = create_enhanced_csv(df_excess_returns)
    output_file_excess = "excess_returns.csv"
    enhanced_df_excess.to_csv(output_file_excess)
    print(f"Excess returns saved to {output_file_excess}")
    
    # Also save clean versions without descriptions
    df_scores.to_csv("strategy_scores_clean.csv")
    df_excess_returns.to_csv("excess_returns_clean.csv")
    print("Clean results saved to strategy_scores_clean.csv and excess_returns_clean.csv")

if __name__ == "__main__":
    main() 