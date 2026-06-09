import numpy as np
import matplotlib.pyplot as plt

# Общие параметры моделирования
SEED = 42
rng = np.random.default_rng(SEED)
FREQUENCY = 915e6          	  # рабочая частота, Гц
C = 3e8                    	  # скорость света, м/с
N_TRIALS = 1000            	  # число испытаний Монте-Карло
DISTANCES = np.arange(1, 11, 1)   # расстояния, м

# Параметры backscatter-радиоканала
params = {
    "P_s": 20,        # мощность источника радиосигнала
    "G_s": 2,         # усиление антенны источника
    "G_d": 0,         # эффективный коэффициент устройства
    "G_r": 2,         # усиление антенны приёмника
    "K_bs": -20,      # коэффициент backscatter-отражения
    "P_sens": -90     # чувствительность приёмника
}

# Параметры сред распространения

environments = {
    "В помещении": {"n": 3.0, "sigma": 6},
    "Уличная среда": {"n": 2.4, "sigma": 4},
    "Промышленная среда": {"n": 3.5,"sigma": 8}
}


# Модели потерь распространения

def free_space_path_loss(distance_m, frequency_hz):
    return 20 * np.log10(4 * np.pi * distance_m * frequency_hz / C)

def path_loss(distance_m, frequency_hz, n, sigma, d0=1.0):
    pl_d0 = free_space_path_loss(d0, frequency_hz)
    shadowing = rng.normal(0, sigma)
    return pl_d0 + 10 * n * np.log10(distance_m / d0) + shadowing


# Расчёт мощности принятого backscatter-сигнала

def received_backscatter_power(d1, d2, frequency_hz, params, env):
    pl_d1 = path_loss(distance_m=d1, frequency_hz=frequency_hz, n=env["n"], sigma=env["sigma"])
    pl_d2 = path_loss(distance_m=d2, frequency_hz=frequency_hz, n=env["n"], sigma=env["sigma"])
    p_rx = (params["P_s"] + params["G_s"] + params["G_d"] + params["G_r"] + params["K_bs"] - pl_d1 - pl_d2)
    return p_rx


# Монте-Карло моделирование вероятности успешной передачи

def simulate_backscatter_probability(distances, params, env, trials=N_TRIALS):
    probabilities = []

    for distance in distances:
        success_count = 0
        for _ in range(trials):
            # Упрощённый симметричный случай:
            # d1 = d2 = d
            p_rx = received_backscatter_power(
                d1=distance,
                d2=distance,
                frequency_hz=FREQUENCY,
                params=params,
                env=env
            )

            if p_rx >= params["P_sens"]:
                success_count += 1

        probabilities.append(success_count / trials)

    return np.array(probabilities)


# Запуск моделирования

results = {}

for env_name, env in environments.items():
    results[env_name] = simulate_backscatter_probability(
        distances=DISTANCES,
        params=params,
        env=env,
        trials=N_TRIALS
    )

# Вывод численных результатов

print("Вероятность успешной backscatter-передачи")
print("-" * 60)
for env_name, probabilities in results.items():
    print(env_name)
    for distance, probability in zip(DISTANCES, probabilities):
        print(f"  d = {distance:2d} м: P_success = {probability:.3f}")
    print()


# Построение графика

plt.figure(figsize=(9, 5))
for env_name, probabilities in results.items():
    plt.plot(DISTANCES,probabilities,label=env_name)

plt.xlabel("Расстояние, м")
plt.ylabel("Вероятность успешной передачи")
plt.title("Вероятность успешной backscatter-передачи от расстояния")
plt.ylim(0, 1.05)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("bs_success_channel_probability.png", dpi=300)
plt.show()