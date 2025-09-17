import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
            res.append(position[1]-position[0])
        return res



def simulate(n_balls, n_levels):
    board = Board(n_balls)

    for _ in range(n_levels):
        board.step()
    experimental_data = board.show()
    # experimental_data = map(lambda x: x+(n_levels//2), experimental_data)

    plt.hist(experimental_data, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    # counts, bin_edges = np.histogram(experimental_data, bins=n_levels, density=True)

    # Midpoints of bins for bar plot
    # bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Plot barplot instead of histogram
    # plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0], alpha=0.6, color='skyblue', edgecolor='black')

    mu=0
    std=n_levels/4

    xmin, xmax = plt.xlim()
    max_range = max(abs(xmin), abs(xmax)) 
    x = np.linspace(-max_range, max_range, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'r', linewidth=2)

    plt.title(f"Distribution comparison")
    plt.xlabel("k")
    plt.ylabel("P[X=k]")

    plt.show()

if __name__ == "__main__":
    
    n=6  # num levels
    N=100000  # num balls

    simulate(n_balls=N, n_levels=n)
