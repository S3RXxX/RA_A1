import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,  binom
import seaborn as sns
import pandas as pd

class Ball:
    
    def __init__(self):
        # we could use only an integer for i and then position would be [i, n_levels-i]
        self.position = [0, 0]
    
    def fall(self):
        event = np.random.randint(0, 2)
        left, right = 1-event, event
        self.position[0] += left
        self.position[1] += right

    def show(self):
        return self.position.copy()


class Board:
    def __init__(self, n_balls):
        self.balls = [Ball() for _ in range(n_balls)]
        # self.__n_balls = n_balls

    def step(self):
        for ball in self.balls:
            ball.fall()
    
    def show(self):
        res = []
        for ball in self.balls:
            position = ball.show()
            # print("position:", position, position[1]-position[0])
            res.append(position[0])
        return res



def simulate(n_balls, n_levels):
    board = Board(n_balls)

    for _ in range(n_levels):
        board.step()
    experimental_data = board.show()

    return experimental_data

def plot(experimental_data, n_levels):

    df = pd.DataFrame({"values": experimental_data})

    # Count frequencies
    freqs = df["values"].value_counts(normalize=True).sort_index().reset_index()
    freqs.columns = ["value", "count"]

    # Barplot
    # sns.barplot(x="value", y="count", data=freqs, color="skyblue", edgecolor="black")
    x_vals = freqs["value"]
    y_vals = freqs["count"]
    plt.bar(x_vals, y_vals, width=0.8, color="skyblue", edgecolor="black", alpha=0.6, label="Experimental (Binomial)")


    mu=n_levels/2
    std=np.sqrt(n_levels/4)
    print(f"mu={mu}; std={std}")
    x = np.linspace(0, n_levels, 10*n_levels)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2)

    plt.title(f"Distribution comparison")
    plt.xlabel("k")
    plt.ylabel("P[X=k]")

    plt.show()

if __name__ == "__main__":
    
    n=20  # num levels
    N=20000  # num balls

    data = simulate(n_balls=N, n_levels=n)
    print(f"min: {min(data)}, max: {max(data)}")
    # print(data)
    plot(data, n)

    ## binomial
    # for r in range(0, n+1):
    #     print(r, ": ", binom.pmf(r, n, 0.5))
