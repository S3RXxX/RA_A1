import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm,  binom, chi2, kstest, anderson
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

def chi2_p(expected_counts, obs_counts):
    while np.any(expected_counts < 5):
        if len(expected_counts) < 2:
            break
        if expected_counts[0] < 5:
            expected_counts[1] += expected_counts[0]
            obs_counts[1] += obs_counts[0]
            expected_counts = expected_counts[1:]
            obs_counts = obs_counts[1:]
        elif expected_counts[-1] < 5:
            expected_counts[-2] += expected_counts[-1]
            obs_counts[-2] += obs_counts[-1]
            expected_counts = expected_counts[:-1]
            obs_counts = obs_counts[:-1]
        else:
            break

    chi2_stat = np.sum((obs_counts - expected_counts)**2 / expected_counts)
    # print(f"CHI2 statistic {chi2_stat}")
    df = len(expected_counts) - 1
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return p_value

def kstest_p(obs_counts, mu, sigma):
    ks_stat, ks_p = kstest(obs_counts, 'norm', args=(mu, sigma))
    # print(f"Kolmogorov-Smirnov statistic {ks_stat}")
    return ks_p


def simulate(n_balls, n_levels):
    board = Board(n_balls)

    for _ in range(n_levels):
        board.step()
    experimental_data = board.show()

    return experimental_data

def plot(experimental_data, n_levels, N_balls, b_plot=True):
    global seed
    
    mu=n_levels/2
    std=np.sqrt(n_levels/4)

    df = pd.DataFrame({"values": experimental_data})


    freqs = df["values"].value_counts(normalize=True).sort_index().reset_index()
    freqs.columns = ["value", "count"]

    all_values = np.arange(0, n_levels + 1)

    full_freqs = pd.DataFrame({"value": all_values}).merge(freqs, on="value", how="left").fillna(0)

    # Barplot
    # sns.barplot(x="value", y="count", data=freqs, color="skyblue", edgecolor="black")
    x_vals = full_freqs["value"]
    y_vals = full_freqs["count"]

    if b_plot:
        plt.bar(x_vals, y_vals, width=0.8, color="skyblue", edgecolor="black", alpha=0.6, label="Experimental (Binomial)")

        granularity = 10
        # print(f"mu={mu}; std={std}")
        x = np.linspace(0, n_levels, granularity*n_levels+1)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'r', linewidth=2)

        plt.title(f"Distribution comparison")
        plt.xlabel("k")
        plt.ylabel("P[X=k]")
        plt.savefig(f"plots/{n_levels}Levels_{N_balls}Balls_{seed}Seed.png")
        plt.close()

    obs_rel = np.array([norm.pdf(c, mu, std) for c in x_vals])

    # mean quadratic error
    tse = 0
    for c in x_vals:
        dif = y_vals[c]-obs_rel[c]
        tse+=(dif)**2
        # print(f"Difference in {c} is {dif}")
    mse = tse/(n_levels+1)
    # print(f"Mean quadratic error: {mse}")

    # CHI2
    expected_counts = y_vals.values * n_levels
    obs_counts = obs_rel * n_levels
    chi2_p_val = chi2_p(expected_counts=expected_counts, obs_counts=obs_counts)

    # Kolmogorov-Smirnov
    ks_p_val = kstest_p(obs_counts=obs_counts, mu=mu, sigma=std)
    # print(f"CHI2 p value = {chi2_p_val}")

    return mse, chi2_p_val, ks_p_val

if __name__ == "__main__":

    # simple execution
    # data = simulate(n_balls=20000, n_levels=20)
    # mse, chi2_, ks_ = plot(data, 20, 20000, b_plot=True)

    # experiments

    seeds = [42, 67, 77, 69, 13]
    n_=[x for x in range(5, 100, 5)]  # num levels  # num levels
    N_=[x for x in range(5,51, 5)]+[x for x in range(60, 101, 10)]+  [x for x in range(150, 501, 50)]+[x for x in range(1000, 10001, 500)]# num balls
    data_ = {"MSE": [], "Chi2pvalue": [], "KSpvalue": [],"n":[], "N":[], "seed": []}
    for n in n_:
        print(n)
        for N in N_:
            for seed in seeds:
                np.random.seed(seed)
                data = simulate(n_balls=N, n_levels=n)
                # print(f"min: {min(data)}, max: {max(data)}")
                # print(data)
                mse, chi2_, ks_ = plot(data, n, N, b_plot=True)
                data_["MSE"].append(mse)
                data_["Chi2pvalue"].append(chi2_)
                data_["KSpvalue"].append(ks_)
                data_["n"].append(n)
                data_["N"].append(N)
                data_["seed"].append(seed)
    data_ = pd.DataFrame(data_)
    data_.to_csv("./data.csv", index=False)

    ## binomial
    # for r in range(0, n+1):
    #     print(r, ": ", binom.pmf(r, n, 0.5))
