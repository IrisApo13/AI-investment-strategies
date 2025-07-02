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
    Process all JSON files in the experiments directory and extract strategy scores.
    
    Returns:
        dict: Dictionary with structure {ticker: {strategy_name: score}}
    """
    results = defaultdict(dict)
    
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
            
            # Get the strategy name and score
            strategy_name = strategies[0].get('name', 'Unknown Strategy')
            score = performance_history[0].get('performance_score', 0)
            
            # Store the result
            results[ticker][strategy_name] = score
            
            print(f"Processed {ticker} - {strategy_name}: {score}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return results

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
    df = df.round(1)
    
    return df

def plot_table(df):
    """
    Create a matplotlib table visualization of the results.
    
    Args:
        df (pandas.DataFrame): The results table
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for the table
    # Add average column
    df_with_avg = df.copy()
    df_with_avg['Average'] = df.mean(axis=1)
    
    # Prepare table data
    table_data = []
    for ticker in df_with_avg.index:
        row = [ticker]
        for col in df_with_avg.columns:
            value = df_with_avg.loc[ticker, col]
            row.append(f"{value:.1f}")
        table_data.append(row)
    
    # Create column headers
    col_labels = ['Stock'] + list(df_with_avg.columns)
    
    # Create the table
    table = ax.table(cellText=table_data, colLabels=col_labels, 
                    cellLoc='center', loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Color code cells based on performance
    for i in range(len(table_data)):
        for j in range(1, len(col_labels)):  # Skip stock name column
            cell = table[(i+1, j)]  # +1 because of header row
            try:
                value = float(table_data[i][j])
                # Color coding: Green for high scores, Red for low scores
                if value >= 60:
                    cell.set_facecolor('#90EE90')  # Light green
                elif value >= 40:
                    cell.set_facecolor('#FFE4B5')  # Light orange
                elif value >= 20:
                    cell.set_facecolor('#FFB6C1')  # Light pink
                else:
                    cell.set_facecolor('#FFCCCB')  # Light red
            except ValueError:
                pass
    
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
    
    Args:
        df (pandas.DataFrame): The results table
    """
    print("\n" + "="*100)
    print("STRATEGY PERFORMANCE SCORES BY COMPANY")
    print("="*100)
    
    if df.empty:
        print("No results found!")
        return
    
    # Create a formatted table with headers
    print(f"{'Stock':<8} | {'Momentum':<10} | {'Price Change':<12} | {'MACD':<8} | {'RSI':<8} | {'Average':<8}")
    print("-" * 100)
    
    # Print each row with formatted data
    for ticker in df.index:
        row = df.loc[ticker]
        avg_score = row.mean()
        
        # Format each score with proper spacing
        momentum = f"{row.get('Momentum Strategy', 0):.1f}" if 'Momentum Strategy' in row else "N/A"
        price_change = f"{row.get('Price Change Strategy', 0):.1f}" if 'Price Change Strategy' in row else "N/A"
        macd = f"{row.get('MACD Strategy', 0):.1f}" if 'MACD Strategy' in row else "N/A"
        rsi = f"{row.get('RSI Strategy', 0):.1f}" if 'RSI Strategy' in row else "N/A"
        
        print(f"{ticker:<8} | {momentum:<10} | {price_change:<12} | {macd:<8} | {rsi:<8} | {avg_score:.1f}")
    
    print("-" * 100)
    
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

def main():
    """Main function to process results and display the table."""
    print("Processing backtest results from experiments directory...")
    
    # Process the files
    results = process_experiment_files()
    
    if not results:
        print("No valid results found!")
        return
    
    # Create the table
    df = create_score_table(results)
    
    # Display the text results
    print_table(df)
    
    # Create matplotlib visualizations
    print("\nCreating matplotlib visualizations...")
    plot_table(df)
    plot_heatmap(df)
    
    # Save enhanced CSV with descriptions
    enhanced_df = create_enhanced_csv(df)
    output_file = "strategy_scores.csv"
    enhanced_df.to_csv(output_file)
    print(f"\nResults saved to {output_file}")
    
    # Also save a clean version without descriptions
    df.to_csv("strategy_scores_clean.csv")
    print("Clean results saved to strategy_scores_clean.csv")

if __name__ == "__main__":
    main() 