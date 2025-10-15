import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,  binom, chi2
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

def chi2_format(expected_counts, obs_counts):
    i = 0
    while np.any(expected_counts < 5):
        if i < len(expected_counts)//2:
            if expected_counts[i] < 5:
                expected_counts[i+1] += expected_counts[i]
                expected_counts = np.delete(expected_counts,i)
                obs_counts = np.delete(obs_counts,i)
                i-=1
        else:
            if expected_counts[i] < 5:
                expected_counts[i-1] += expected_counts[i]
                expected_counts = np.delete(expected_counts,i)
                obs_counts = np.delete(obs_counts,i)
                i -= 1
        i+=1

        # if expected_counts[0] < 5:
        #     expected_counts[1] += expected_counts[0]
        #     obs_counts[1] += obs_counts[0]
        #     expected_counts = expected_counts[1:]
        #     obs_counts = obs_counts[1:]
        # elif expected_counts[-1] < 5:
        #     expected_counts[-2] += expected_counts[-1]
        #     obs_counts[-2] += obs_counts[-1]
        #     expected_counts = expected_counts[:-1]
        #     obs_counts = obs_counts[:-1]
        # else:
        #     break

    chi2_stat = np.sum((obs_counts - expected_counts)**2 / expected_counts)
    print(f"CHI2 statistic {chi2_stat}")
    df = len(expected_counts) - 1 - 2  # minus 2 for estimated mean and std
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return p_value

def simulate(n_balls, n_levels):
    board = Board(n_balls)

    for _ in range(n_levels):
        board.step()
    experimental_data = board.show()

    return experimental_data

def plot(experimental_data, n_levels, N_balls, b_plot=True):

    mu=n_levels/2
    std=np.sqrt(n_levels/4)

    df = pd.DataFrame({"values": experimental_data})

    # Count frequencies
    freqs = df["values"].value_counts(normalize=True).sort_index().reset_index()
    freqs.columns = ["value", "count"]

    # Create full range of possible values (0..n_levels)
    all_values = np.arange(0, n_levels + 1)

    # Merge with all possible values, filling missing frequencies with 0
    full_freqs = pd.DataFrame({"value": all_values}).merge(freqs, on="value", how="left").fillna(0)

    # Barplot
    # sns.barplot(x="value", y="count", data=freqs, color="skyblue", edgecolor="black")
    x_vals = full_freqs["value"]
    y_vals = full_freqs["count"]

    if b_plot:
        plt.bar(x_vals, y_vals, width=0.8, color="skyblue", edgecolor="black", alpha=0.6, label="Experimental (Binomial)")

        granularity = 10
        print(f"mu={mu}; std={std}")
        x = np.linspace(0, n_levels, granularity*n_levels+1)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'r', linewidth=2)

        plt.title(f"Distribution comparison")
        plt.xlabel("k")
        plt.ylabel("P[X=k]")
        plt.savefig(f"plots/{n_levels}Levels_{N_balls}Balls.png")
        plt.close()

    obs_rel = np.array([norm.pdf(c, mu, std) for c in x_vals])

    # mean quadratic error
    tse = 0
    for c in x_vals:
        dif = y_vals[c]-obs_rel[c]
        tse+=(dif)**2
        # print(f"Difference in {c} is {dif}")
    mse = tse/(n_levels+1)
    print(f"Mean quadratic error: {mse}")

    # performa CHI^2
    expected_counts = y_vals.values * n_levels
    obs_counts = obs_rel * n_levels
    p_val = chi2_format(expected_counts=expected_counts, obs_counts=obs_counts)
    print(f"p value = {p_val}")
    return mse, p_val

if __name__ == "__main__":
    
    n_=[20,80,120,300,500]  # num levels
    N_=[10,100,500,1000,10000,20000]  # num balls
    data_ = {"MSE": [], "Chi2pvalue": [], "n":[], "N":[]}
    for n in n_:
        for N in N_:
            data = simulate(n_balls=N, n_levels=n)
            print(f"min: {min(data)}, max: {max(data)}")
            # print(data)
            mse, chi2_ = plot(data, n, N, b_plot=True)
            data_["MSE"].append(mse)
            data_["Chi2pvalue"].append(chi2_)
            data_["n"].append(n)
            data_["N"].append(N)
    data_ = pd.DataFrame(data_)
    data_.to_csv("./data.csv", index=False)

    ## binomial
    # for r in range(0, n+1):
    #     print(r, ": ", binom.pmf(r, n, 0.5))
