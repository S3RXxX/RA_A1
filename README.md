# Galton Board Simulation and Distribution Comparison

This project simulates a **Galton board** to study how the experimental **binomial distribution** of falling balls approaches a **normal distribution** as the number of levels increases.  
It then compares the two distributions using:
- **Chi-squared goodness-of-fit test**
- Mean Squared Error (MSE)

and generates plots of the resulting distributions.

---

## Project structure
simulate.py # main simulation script, it creates data.csv

plots/ # folder where we save the plots that compare the sampled distribution with the theoretical

plot_together/ # folder where we save different plots that relates number of levels, number of balls and RMSE*100

data/ # folder were we save different plots that relates number of levels, number of balls and RMSE*100 but fixing levels or balls.

balls_fixedChi/ # same as data/ but we instead of RMSE*100 with Chi^2 pvalue and fixing the number of balls.

n_fixedChi/ # same as balls_fixedChi/ but fixing the number of levels.

data.csv # file where results (MSE, Chi²) are saved

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



