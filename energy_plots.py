import numpy as np
import matplotlib.pyplot as plt

from energy_model import collect_plot_data, HOURS, TX_INTERVAL_MIN

N = 1000
SEED = 42

plot_data = collect_plot_data(n=N, seed=SEED)

sources = plot_data["sources_order"]
environments = plot_data["environments_order"]

autonomy_probability = plot_data["autonomy_probability"]
successful_tx = plot_data["successful_tx"]

max_tx = HOURS * 60 // TX_INTERVAL_MIN


# График вероятности автономной работы

def plot_autonomy_probability():
    x = np.arange(len(sources))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, env in enumerate(environments):
        values = autonomy_probability[env]
        positions = x + (i - 1) * width

        bars = ax.bar(positions, values, width, label=env)

        for bar, value in zip(bars, values):
            x_pos = bar.get_x() + bar.get_width() / 2
            bar_color = bar.get_facecolor()

            if value == 0:
                y_pos = 2.0
                label = "0%"
            elif value < 5:
                y_pos = 4.0
                label = f"{value:.1f}%"
            else:
                y_pos = value + 2
                label = f"{value:.1f}%"

            ax.text(x_pos,y_pos,label,ha="center",va="bottom",fontsize=8,color=bar_color,fontweight="bold")

    ax.set_xlabel("Источник энергии")
    ax.set_ylabel("Вероятность автономной работы, %")
    ax.set_title("Вероятность автономной работы Ambient IoT-устройства\nпри разных источниках энергии")
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y")
    ax.legend(loc="upper center")

    plt.tight_layout()
    plt.savefig("autonomy_probability.png", dpi=300)
    plt.show()


# График среднего числа успешных передач

def plot_successful_transmissions():
    x = np.arange(len(sources))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, env in enumerate(environments):
        values = successful_tx[env]
        positions = x + (i - 1) * width

        bars = ax.bar(positions, values, width, label=env)

        for bar, value in zip(bars, values):
            x_pos = bar.get_x() + bar.get_width() / 2
            bar_color = bar.get_facecolor()

            ax.text(x_pos,value + 3,f"{value:.1f}",ha="center",va="bottom",fontsize=8,color=bar_color,fontweight="bold")

    ax.set_xlabel("Источник энергии")
    ax.set_ylabel("Среднее число успешных передач")
    ax.set_title("Среднее число успешных передач Ambient IoT-устройства\nпри разных источниках энергии")
    ax.set_xticks(x)
    ax.set_xticklabels(sources)
    ax.set_ylim(0, max_tx + 15)
    ax.grid(True, axis="y")
    ax.legend(loc="upper center")

    plt.tight_layout()
    plt.savefig("successful_transmissions.png", dpi=300)
    plt.show()



# Запуск

if __name__ == "__main__":
    plot_autonomy_probability()
    plot_successful_transmissions()