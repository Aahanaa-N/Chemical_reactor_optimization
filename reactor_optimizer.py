# Chemical Reactor Optimization Project
# CSTR + PFR Reactor System Profit Maximization

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

print(" Chemical Reactor Optimization Project")
print("=========================================")

# ============================================================================
# 1. DEFINE REACTION PARAMETERS and ECONOMIC CONSTANTS
# ============================================================================
print("Setting up reaction parameters...")

# Reaction: A -> B (First-order reaction)
k = 0.5  # Rate constant [1/min]

# Flow conditions
C_A0 = 1.0      # Initial concentration of A [mol/L]
F_A0 = 10.0     # Molar flow rate of A into the process [mol/min]

# Economic parameters
price_B = 10.0  # Price of product B [$/mol]
price_A = 2.0   # Cost of reactant A [$/mol]

# Reactor costs: cost = a * (Volume)^b
cost_factor_CSTR = 5.0  # [$/(L)^0.6]
cost_factor_PFR = 3.0   # [$/(L)^0.6]
cost_exponent = 0.6

print(" Parameters loaded successfully!")

# ============================================================================
# 2. REACTOR MODEL FUNCTIONS
# ============================================================================
def cstr_solver(V_cstr, F_A0, k, C_A0):
    """
    Solves the CSTR design equation for a first-order reaction A -> B.
    Returns the conversion X achieved in the CSTR.
    """
    # For a 1st order reaction: X = (k * tau) / (1 + k * tau)
    tau = V_cstr * C_A0 / F_A0  # Space time
    X_cstr = (k * tau) / (1 + k * tau)
    return X_cstr

def pfr_ode(X, V, F_A0, k, C_A0):
    """
    The differential equation for a PFR: dX/dV = (-r_A) / F_A0
    For 1st order: dX/dV = k * C_A0 * (1 - X) / F_A0
    """
    dXdV = (k * C_A0 / F_A0) * (1 - X)
    return dXdV

def pfr_solver(X_inlet, V_pfr, F_A0, k, C_A0):
    """
    Solves the PFR by integrating the dX/dV equation.
    Returns the final conversion at the exit of the PFR.
    """
    V_span = np.linspace(0, V_pfr, 50)  # Volume points to integrate over
    X_solution = odeint(pfr_ode, X_inlet, V_span, args=(F_A0, k, C_A0))
    X_final = X_solution[-1, 0]  # Get the last value (conversion at exit)
    return X_final

print(" Reactor models defined successfully!")

# ============================================================================
# 3. THE PROFIT FUNCTION (The function we want to MAXIMIZE)
# ============================================================================
def profit_function(variables):
    """
    Calculates the profit for a given set of reactor volumes.
    We MINIMIZE the negative of this function to MAXIMIZE profit.
    variables = [V_cstr, V_pfr]
    """
    V_cstr, V_pfr = variables

    # Avoid negative volumes (causes math errors)
    if V_cstr < 0 or V_pfr < 0:
        return 1e9  # Return a huge, bad value if volumes are negative

    # 1. Calculate conversion from CSTR
    X_cstr = cstr_solver(V_cstr, F_A0, k, C_A0)

    # 2. Calculate final conversion from PFR
    X_final = pfr_solver(X_cstr, V_pfr, F_A0, k, C_A0)

    # 3. ECONOMIC CALCULATION
    # Income: Moles of B produced per minute * price of B
    income = F_A0 * X_final * price_B

    # Cost of Raw Material
    raw_material_cost = F_A0 * price_A

    # Capital Cost of Reactors (amortized per minute)
    minutes_per_year = 365 * 24 * 60
    total_minutes_lifetime = 3 * minutes_per_year

    capital_cost_cstr = (cost_factor_CSTR * (V_cstr ** cost_exponent)) / total_minutes_lifetime
    capital_cost_pfr = (cost_factor_PFR * (V_pfr ** cost_exponent)) / total_minutes_lifetime

    # Total Profit per minute
    total_profit_per_min = income - raw_material_cost - capital_cost_cstr - capital_cost_pfr

    # Return NEGATIVE profit because 'minimize' finds minimum
    return -total_profit_per_min

print(" Profit function defined successfully!")

# ============================================================================
# 4. OPTIMIZATION and MAIN SCRIPT
# ============================================================================
print("\nStarting optimization process...")
print("This may take a few seconds...")

# Initial guess for the optimizer [V_cstr_initial, V_pfr_initial]
initial_guess = [10.0, 20.0]  # Liters

# Set bounds for the variables (prevent negative volumes)
bounds = [(1e-5, 1000), (1e-5, 1000)]

# Run the optimization
result = minimize(profit_function, initial_guess, bounds=bounds, method='L-BFGS-B')

# Extract the optimal values
V_cstr_opt, V_pfr_opt = result.x
max_profit = -result.fun  # Remember we minimized the negative profit

# Calculate conversions at optimum
X_cstr_opt = cstr_solver(V_cstr_opt, F_A0, k, C_A0)
X_final_opt = pfr_solver(X_cstr_opt, V_pfr_opt, F_A0, k, C_A0)

# ============================================================================
# 5. DISPLAY RESULTS
# ============================================================================
print("\n" + "="*60)
print(" OPTIMIZATION RESULTS")
print("="*60)
print(f"Optimal CSTR Volume: {V_cstr_opt:.2f} L")
print(f"Optimal PFR Volume:  {V_pfr_opt:.2f} L")
print(f"Maximum Profit: ${max_profit:.4f} per minute")
print(f"Conversion after CSTR: {X_cstr_opt:.3f} ({X_cstr_opt*100:.1f}%)")
print(f"Final Conversion after PFR: {X_final_opt:.3f} ({X_final_opt*100:.1f}%)")
print(f"\nOptimization successful: {result.success}")

# ============================================================================
# 6. VISUALIZATION (Bonus!)
# ============================================================================
print("\n Generating profit visualization...")

# Create a simple visualization
import matplotlib.pyplot as plt

# Test different CSTR volumes to see profit trend
V_cstr_test = np.linspace(1, 50, 20)
profits = []

for V_cstr in V_cstr_test:
    # For each CSTR volume, find best PFR volume
    test_result = minimize(lambda V_pfr: profit_function([V_cstr, V_pfr[0]]), 
                          [20.0], bounds=[(1e-5, 1000)], method='L-BFGS-B')
    profits.append(-test_result.fun)

plt.figure(figsize=(10, 6))
plt.plot(V_cstr_test, profits, 'b-', linewidth=2, label='Maximum Profit')
plt.axvline(V_cstr_opt, color='red', linestyle='--', label=f'Optimal CSTR Volume: {V_cstr_opt:.1f} L')
plt.xlabel('CSTR Volume (L)')
plt.ylabel('Profit per Minute ($)')
plt.title('Profit vs CSTR Volume (with Optimal PFR for each CSTR)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n Analysis complete! Check the graph above to see how profit changes with reactor size.")
