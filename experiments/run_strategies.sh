#!/bin/bash

# Define arrays for companies and strategies
companies=("TSLA" "AAPL" "GOOG" "MSFT")
strategies=("rsi_strategy.json" "macd_strategy.json" "price_change_strategy.json" "momentum_strategy.json")

# Loop through each company
for company in "${companies[@]}"; do
    echo "Testing strategies for $company..."
    
    # Loop through each strategy
    for strategy in "${strategies[@]}"; do
        echo "Running $strategy for $company..."
        python ../main.py "$company" --strategy-file "../strategies/$strategy" --iterations 1
        echo "Completed $strategy for $company"
        echo "----------------------------------------"
    done
    
    echo "Completed all strategies for $company"
    echo "========================================"
done

echo "All backtesting completed!"

