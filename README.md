# Galton Box Simulation and Distribution Comparison

This project simulates a **Galton board (bean machine)** to study how the experimental **binomial distribution** of falling balls approaches a **normal distribution** as the number of levels increases.  
It then compares the two distributions using:
- **Chi-squared goodness-of-fit test**
- **Kolmogorov–Smirnov (K–S) test**
- Mean Squared Error (MSE)

and generates plots of the resulting distributions.

---

## Project structure
simulate.py # main simulation script
plots/ # folder where generated plots are saved
data.csv # file where results (MSE, Chi², KS p-values) are saved
requirements.txt # list of Python dependencies
README.md # this file


---


## ⚙️ Installation

Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate       # on macOS/Linux
venv\Scripts\activate          # on Windows
pip install -r requirements.txt
python simulate.py
```



