#!/bin/bash

# Configuration
#Set the number of times to run each test
NUM_RUNS=4

# Define arrays for companies and strategies
companies=("TSLA" "AAPL" "GOOG" "MSFT" "INTC" "UNH")

echo "Starting AI backtesting with $NUM_RUNS runs per test..."
echo "Total tests to run: $(( ${#companies[@]} * ${#strategies[@]} * NUM_RUNS ))"
echo "========================================"

# Loop through each company
for company in "${companies[@]}"; do
    echo "Testing strategies for $company..."
    
    # Loop through each strategy
        # Run the test multiple times
    for run in $(seq 1 $NUM_RUNS); do
        echo "  Run $run/$NUM_RUNS..."
        python ../main.py "$company" --iterations 1
            
        # Add a small delay between runs to avoid overwhelming the system
        if [ $run -lt $NUM_RUNS ]; then
            sleep 2
        fi
    done
        
    echo "Completed all $NUM_RUNS runs of $strategy for $company"
    echo "----------------------------------------"

done
    

echo "All backtesting completed!"
echo "Total runs completed: $(( ${#companies[@]} * ${#strategies[@]} * NUM_RUNS ))"

