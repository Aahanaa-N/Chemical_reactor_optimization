# Chemical Reactor Optimization 

## Project Overview
This Python project demonstrates the optimization of a chemical reactor system consisting of a Continuous Stirred-Tank Reactor (CSTR) followed by a Plug Flow Reactor (PFR). The goal is to maximize profit by finding the optimal reactor volumes that balance capital costs against product revenue for a first-order chemical reaction A → B.

##  Chemical Engineering Concepts
- **Reaction Kinetics**: First-order reaction A → B
- **Reactor Design**: CSTR and PFR mole balances
- **Process Economics**: Capital vs. operational cost optimization
- **Vapor-Liquid Equilibrium**: Mass and energy balances
- **Numerical Methods**: Optimization and differential equation solving

## Features
- Optimizes reactor volumes for maximum profitability
- Models chemical reaction kinetics and conversion rates
- Considers both capital and raw material costs
- Includes visualization of profit vs. reactor size relationships
- Easily customizable parameters for different scenarios

## Results
The program determines:
- Optimal CSTR and PFR volumes
- Maximum achievable profit per minute
- Conversion rates at each reactor stage
- Sensitivity analysis through parameter variation

## Technologies Used
- **Python** - Core programming language
- **NumPy** - Numerical computations
- **SciPy** - Optimization and integration algorithms
- **Matplotlib** - Data visualization

## Installation & Usage
```bash
# Clone the repository
git clone https://github.com/your-username/chemical-reactor-optimization.git

# Install required packages
pip install numpy scipy matplotlib

# Run the optimization
python reactor_optimizer.py
