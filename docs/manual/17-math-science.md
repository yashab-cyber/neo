# Chapter 17: Mathematics & Science
**Advanced Problem Solving Across STEM Domains**

---

## 17.1 Overview of Mathematical Capabilities

NEO's mathematical intelligence combines symbolic computation, numerical analysis, and advanced problem-solving techniques to handle complex mathematical challenges across all levels, from basic arithmetic to graduate-level research mathematics.

### Core Mathematical Domains
- **Algebra**: Linear algebra, abstract algebra, polynomial systems
- **Calculus**: Differential and integral calculus, multivariable calculus
- **Statistics**: Descriptive and inferential statistics, hypothesis testing
- **Discrete Mathematics**: Graph theory, combinatorics, number theory
- **Applied Mathematics**: Optimization, differential equations, numerical methods
- **Pure Mathematics**: Real analysis, complex analysis, topology

## 17.2 Basic Mathematical Operations

### Arithmetic and Basic Algebra
```bash
# Basic calculations
neo "Calculate 2^10 * 15 + sqrt(144)"
neo "Solve for x: 3x + 7 = 22"
neo "Simplify (x^2 - 4) / (x - 2)"

# Fraction operations
neo "Add 2/3 + 5/8"
neo "Simplify 45/60"

# Percentage calculations
neo "What is 15% of 240?"
neo "If 25 is 20% of a number, what is the number?"
```

### Advanced Algebraic Operations
```bash
# Polynomial operations
neo "Expand (x + 3)(x^2 - 2x + 5)"
neo "Factor x^3 - 8"
neo "Find the roots of x^2 - 5x + 6 = 0"

# System of equations
neo "Solve the system: 2x + 3y = 7, x - y = 1"
neo "Solve: x + y + z = 6, 2x - y + 3z = 14, x + 2y - z = 2"

# Matrix operations
neo "Multiply matrices [[1,2],[3,4]] and [[5,6],[7,8]]"
neo "Find the inverse of matrix [[2,1],[3,2]]"
neo "Calculate the determinant of [[1,2,3],[4,5,6],[7,8,9]]"
```

## 17.3 Calculus and Analysis

### Differential Calculus
```bash
# Derivatives
neo "Find the derivative of x^3 + 2x^2 - 5x + 1"
neo "Differentiate sin(x^2) with respect to x"
neo "Find the partial derivatives of f(x,y) = x^2y + xy^2"

# Applications of derivatives
neo "Find the critical points of f(x) = x^3 - 3x^2 + 2"
neo "Determine the maximum and minimum of f(x) = x^2 - 4x + 3 on [0,5]"
neo "Find the equation of the tangent line to y = x^2 at x = 3"
```

### Integral Calculus
```bash
# Integration
neo "Integrate x^2 + 3x - 2 dx"
neo "Evaluate the definite integral of sin(x) from 0 to π"
neo "Find the area between y = x^2 and y = 4"

# Advanced integration techniques
neo "Integrate by parts: ∫ x*e^x dx"
neo "Use substitution to evaluate ∫ 2x*sqrt(x^2 + 1) dx"
neo "Integrate 1/(x^2 + 1) using trigonometric substitution"
```

### Multivariable Calculus
```bash
# Partial derivatives and gradients
neo "Find the gradient of f(x,y) = x^2 + y^2"
neo "Calculate the directional derivative of f(x,y) = xy in direction (1,1)"

# Multiple integrals
neo "Evaluate the double integral ∫∫ xy dA over the region R = [0,2] × [0,3]"
neo "Find the volume under z = x^2 + y^2 over the square [0,1] × [0,1]"
```

## 17.4 Statistics and Probability

### Descriptive Statistics
```bash
# Basic statistics
neo "Calculate mean, median, mode of: 12, 15, 18, 12, 20, 25, 12"
neo "Find the standard deviation of the dataset: 5, 7, 9, 11, 13"
neo "Calculate the correlation coefficient between X and Y" --data="file.csv"

# Data visualization
neo "Create a histogram for the data: 1,2,2,3,3,3,4,4,5"
neo "Generate a box plot for the sales data" --input="sales.csv"
neo "Plot a scatter diagram for height vs weight data"
```

### Probability Theory
```bash
# Basic probability
neo "What's the probability of getting exactly 3 heads in 5 coin flips?"
neo "Calculate P(X = 4) where X follows Poisson distribution with λ = 2"
neo "Find the probability that a normal random variable with μ=100, σ=15 is between 85 and 115"

# Probability distributions
neo "Plot the PDF of a normal distribution with mean 0 and standard deviation 1"
neo "Calculate the CDF of exponential distribution with rate parameter 2 at x = 1"
```

### Inferential Statistics
```bash
# Hypothesis testing
neo "Perform a t-test to compare means of two groups" --data="groups.csv"
neo "Test if the population mean equals 50 with sample mean 52, std 8, n=25"
neo "Conduct a chi-square test for independence" --data="contingency.csv"

# Confidence intervals
neo "Calculate 95% confidence interval for population mean" --sample-data="values.txt"
neo "Find confidence interval for difference in proportions"
```

## 17.5 Linear Algebra

### Vector Operations
```bash
# Vector calculations
neo "Calculate dot product of vectors [1,2,3] and [4,5,6]"
neo "Find cross product of [1,0,2] and [3,1,0]"
neo "Normalize the vector [3,4,5]"
neo "Find the angle between vectors [1,1] and [1,-1]"
```

### Matrix Operations
```bash
# Advanced matrix operations
neo "Find eigenvalues and eigenvectors of [[4,1],[2,3]]"
neo "Perform LU decomposition of matrix [[2,1,3],[4,3,1],[6,2,4]]"
neo "Calculate the Moore-Penrose pseudoinverse"
neo "Solve the least squares problem Ax = b" --matrix-file="A.csv" --vector-file="b.csv"
```

### Applications
```bash
# Real-world applications
neo "Solve the Markov chain with transition matrix P" --matrix="transitions.csv"
neo "Perform Principal Component Analysis on dataset" --data="features.csv"
neo "Find the best fit line using linear regression"
```

## 17.6 Discrete Mathematics

### Combinatorics
```bash
# Counting problems
neo "How many ways can you arrange 5 people in a row?"
neo "Calculate C(10,3) - combinations of 10 things taken 3 at a time"
neo "How many ways to distribute 12 identical balls into 4 distinct boxes?"

# Advanced combinatorics
neo "Find the number of derangements of 6 objects"
neo "Calculate the Catalan number C_5"
neo "Solve the inclusion-exclusion principle problem: |A ∪ B ∪ C|"
```

### Graph Theory
```bash
# Graph algorithms
neo "Find the shortest path in the graph" --graph="network.json"
neo "Determine if the graph is planar"
neo "Calculate the chromatic number of the graph"
neo "Find the minimum spanning tree using Kruskal's algorithm"

# Graph properties
neo "Check if the graph is bipartite"
neo "Find all connected components"
neo "Calculate the diameter of the graph"
```

### Number Theory
```bash
# Number theory problems
neo "Find the GCD of 48 and 18"
neo "Calculate 3^17 mod 7"
neo "Factor the number 1247 into prime factors"
neo "Solve the Diophantine equation 3x + 5y = 1"

# Advanced number theory
neo "Find all solutions to x^2 ≡ 1 (mod 8)"
neo "Calculate the Euler totient function φ(100)"
neo "Test if 1009 is prime"
```

## 17.7 Physics Applications

### Classical Mechanics
```bash
# Kinematics
neo "A ball is thrown upward with initial velocity 20 m/s. When does it hit the ground?"
neo "Calculate the trajectory of a projectile with initial velocity 30 m/s at 45°"
neo "Find the acceleration of an object with position function s(t) = 3t^2 + 2t"

# Dynamics
neo "Calculate the force needed to accelerate a 5kg mass at 2 m/s²"
neo "Find the tension in a rope supporting a 10kg mass"
neo "Solve for the motion of a pendulum with length 1m"
```

### Thermodynamics
```bash
# Heat and energy
neo "Calculate the heat required to raise 2kg of water from 20°C to 100°C"
neo "Find the efficiency of a Carnot engine operating between 400K and 300K"
neo "Determine the pressure of an ideal gas with given volume and temperature"
```

### Electromagnetism
```bash
# Electric fields and circuits
neo "Calculate the electric field at distance r from a point charge Q"
neo "Find the resistance of resistors in parallel: 4Ω, 6Ω, 12Ω"
neo "Determine the magnetic force on a current-carrying wire"
```

## 17.8 Chemistry Calculations

### Stoichiometry
```bash
# Chemical equations
neo "Balance the equation: C₂H₆ + O₂ → CO₂ + H₂O"
neo "Calculate moles of CO₂ produced from 5.6g of C₂H₆"
neo "Find the limiting reagent in: 2A + 3B → C, given 4 moles A and 5 moles B"

# Solution chemistry
neo "Calculate molarity of 2.5g NaCl in 500mL solution"
neo "Find the pH of 0.1M HCl solution"
neo "Determine the concentration after dilution: 100mL of 2M to 500mL total"
```

### Thermochemistry
```bash
# Energy calculations
neo "Calculate ΔH for reaction using bond energies"
neo "Find the heat of combustion for methane"
neo "Determine equilibrium constant from ΔG = -25 kJ/mol"
```

## 17.9 Biology and Life Sciences

### Population Dynamics
```bash
# Population models
neo "Model exponential growth with initial population 1000, growth rate 5%"
neo "Calculate carrying capacity from logistic growth equation"
neo "Solve predator-prey equations with given parameters"

# Genetics calculations
neo "Calculate probability of offspring genotype from Punnett square"
neo "Determine allele frequencies using Hardy-Weinberg equilibrium"
neo "Find the probability of genetic inheritance patterns"
```

### Biochemistry
```bash
# Enzyme kinetics
neo "Calculate Km and Vmax from Michaelis-Menten data"
neo "Determine inhibition type from kinetic data"
neo "Find the pH at half-equivalence point in titration"
```

## 17.10 Engineering Mathematics

### Signal Processing
```bash
# Fourier analysis
neo "Calculate the Fourier transform of f(t) = e^(-t²)"
neo "Find the frequency components of the signal" --data="signal.wav"
neo "Apply low-pass filter with cutoff frequency 1000 Hz"
```

### Control Systems
```bash
# Control theory
neo "Calculate the transfer function of the system"
neo "Find the poles and zeros of H(s) = (s+1)/(s²+2s+1)"
neo "Determine system stability using Routh-Hurwitz criterion"
neo "Design PID controller for given plant"
```

### Optimization
```bash
# Optimization problems
neo "Minimize f(x,y) = x² + y² subject to x + y = 1"
neo "Solve linear programming problem" --constraints="constraints.txt"
neo "Find the optimal solution using gradient descent"
neo "Apply genetic algorithm to optimization problem"
```

## 17.11 Research-Level Mathematics

### Advanced Analysis
```bash
# Real and complex analysis
neo "Prove that the sequence converges using epsilon-delta definition"
neo "Find the Laurent series expansion of f(z) = 1/(z(z-1))"
neo "Calculate the residue at each pole"
neo "Evaluate the contour integral using residue theorem"
```

### Abstract Algebra
```bash
# Group theory
neo "Determine if the set forms a group under the given operation"
neo "Find all subgroups of the symmetric group S₄"
neo "Calculate the order of the element in the group"
neo "Prove that the homomorphism is well-defined"
```

### Topology
```bash
# Topological concepts
neo "Determine if the space is compact"
neo "Find the fundamental group of the space"
neo "Check if the function is continuous at the point"
neo "Calculate the homology groups of the complex"
```

## 17.12 Computational Mathematics

### Numerical Methods
```bash
# Numerical analysis
neo "Solve the differential equation y' = y using Euler's method"
neo "Find the root of f(x) = x³ - 2 using Newton's method"
neo "Approximate the integral using Simpson's rule"
neo "Solve the system of linear equations using Gaussian elimination"
```

### Mathematical Modeling
```bash
# Model development
neo "Create a mathematical model for epidemic spread"
neo "Model the cooling of an object using Newton's law"
neo "Develop a queuing theory model for customer service"
neo "Build a financial model for option pricing"
```

## 17.13 Visualization and Graphing

### 2D and 3D Plotting
```bash
# Graph generation
neo "Plot the function f(x) = sin(x) + cos(2x) from 0 to 2π"
neo "Create a 3D surface plot of z = x² + y²"
neo "Generate a parametric plot of x(t) = cos(t), y(t) = sin(t)"
neo "Plot the vector field F(x,y) = (y, -x)"

# Statistical plots
neo "Create a probability distribution plot for normal(μ=0, σ=1)"
neo "Generate a phase portrait for the differential equation system"
neo "Plot the convergence of the numerical method"
```

### Interactive Visualization
```bash
# Dynamic plots
neo "Create an interactive plot showing parameter effects"
neo "Generate animation of function transformation"
neo "Build interactive 3D model of molecular structure"
neo "Create dynamic visualization of algorithm execution"
```

---

**Next Chapter**: [Data Analysis](18-data-analysis.md)

*NEO's mathematical capabilities transform complex problems into manageable solutions, supporting both educational exploration and cutting-edge research across all STEM disciplines.*
