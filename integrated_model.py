import numpy as np
import matplotlib.pyplot as plt

# Общие параметры моделирования
SEED = 42
N_MC = 1000
DT_MIN = 1
HOURS = 48
STEPS = HOURS * 60 // DT_MIN
TIME_MIN = np.arange(STEPS)
TIME_H = TIME_MIN / 60.0
MEASURE_INTERVAL_MIN = 5
TX_INTERVAL_MIN = 15
MAX_TX = HOURS * 60 // TX_INTERVAL_MIN


# Параметры Ambient IoT-устройства
VDD = 3.0
P_SLEEP = (0.94e-6 + 0.30e-6) * VDD     
P_ACTIVE = 3.4e-3 * VDD                
P_TX = 7.0e-3 * VDD                     
T_MEASURE = 0.5                      
T_TX = 0.1                            
E_SLEEP = P_SLEEP * 60.0               
E_CAP_DEFAULT = 1.8                    


# Параметры RF-сбора энергии
PT_RF_W = 1.0                          
GT_RF = 1.0                             
GR_RF = 1.0                             
F_RF_HARVEST = 2.4e9                    
C = 3e8                                 
ETA_RF = 0.5
LAMBDA_RF_HARVEST = C / F_RF_HARVEST



# Параметры backscatter-радиоканала
F_BACKSCATTER = 915e6                   
CHANNEL_PARAMS = {
    "P_s": 20,       # мощность источника радиосигнала
    "G_s": 2,        # усиление антенны источника
    "G_d": 0,        # эффективный коэффициент устройства
    "G_r": 2,        # усиление антенны приёмника
    "K_bs": -20,     # коэффициент backscatter-отражения
    "P_sens": -90    # чувствительность приёмника
}


# Среды эксплуатации

ENVIRONMENTS = {
    "В помещении": {
        "rf_distance": 4.0,
        "path_loss_n": 3.0,
        "shadow_sigma": 6
    },
    "Уличная среда": {
        "rf_distance": 12.0,
        "path_loss_n": 2.4,
        "shadow_sigma": 4
    },
    "Промышленная среда": {
        "rf_distance": 3.0,
        "path_loss_n": 3.5,
        "shadow_sigma": 8
    }
}



# Генерация случайных параметров Монте-Карло

def sample_mc_params(env_name, n, rng):
    base_d = ENVIRONMENTS[env_name]["rf_distance"]

    return {
        # расстояние до RF-источника для сбора энергии
        "d_rf": np.clip(
            base_d * rng.lognormal(0.0, 0.35, n),
            0.5,
            25.0
        ),

        # начальная энергия накопителя
        "e_init": rng.uniform(
            0.2 * E_CAP_DEFAULT,
            E_CAP_DEFAULT,
            n
        ),

        # максимальная энергия накопителя
        "e_cap": np.clip(
            rng.normal(E_CAP_DEFAULT, 0.35 * E_CAP_DEFAULT, n),
            0.5 * E_CAP_DEFAULT,
            3.0 * E_CAP_DEFAULT
        ),

        # мощность передачи
        "p_tx": np.clip(
            P_TX * rng.lognormal(0.0, 0.25, n),
            0.4 * P_TX,
            3.0 * P_TX
        ),

        # мощность активного режима
        "p_active": np.clip(
            P_ACTIVE * rng.lognormal(0.0, 0.15, n),
            0.6 * P_ACTIVE,
            2.0 * P_ACTIVE
        )
    }


# RF-сбор энергии

def rf_harvested_power(params):
    p_h = (
        ETA_RF
        * PT_RF_W
        * GT_RF
        * GR_RF
        * (LAMBDA_RF_HARVEST / (4.0 * np.pi * params["d_rf"])) ** 2
    )

    return p_h


# Модель backscatter-радиоканала

def free_space_path_loss(distance_m, frequency_hz):
    return 20 * np.log10(4 * np.pi * distance_m * frequency_hz / C)


def path_loss(distance_m, frequency_hz, path_loss_n, shadow_sigma, rng, d0=1.0):
    pl_d0 = free_space_path_loss(d0, frequency_hz)
    shadowing = rng.normal(0.0, shadow_sigma, size=np.shape(distance_m))

    return (
        pl_d0
        + 10 * path_loss_n * np.log10(distance_m / d0)
        + shadowing
    )


def received_backscatter_power(distance_m, env, rng):
    pl_1 = path_loss(
        distance_m=distance_m,
        frequency_hz=F_BACKSCATTER,
        path_loss_n=env["path_loss_n"],
        shadow_sigma=env["shadow_sigma"],
        rng=rng
    )

    pl_2 = path_loss(
        distance_m=distance_m,
        frequency_hz=F_BACKSCATTER,
        path_loss_n=env["path_loss_n"],
        shadow_sigma=env["shadow_sigma"],
        rng=rng
    )

    p_rx_bs = (
        CHANNEL_PARAMS["P_s"]
        + CHANNEL_PARAMS["G_s"]
        + CHANNEL_PARAMS["G_d"]
        + CHANNEL_PARAMS["G_r"]
        + CHANNEL_PARAMS["K_bs"]
        - pl_1
        - pl_2
    )

    return p_rx_bs


# Интегрированное моделирование

def simulate_integrated_rf_backscatter(
    env_name,
    distance_m,
    n=N_MC,
    seed=SEED
):
    rng = np.random.default_rng(seed)

    env = ENVIRONMENTS[env_name]
    params = sample_mc_params(env_name, n, rng)

    # RF-мощность сбора энергии для каждого прогона
    p_h = rf_harvested_power(params)

    e_store = params["e_init"].copy()
    e_measure = params["p_active"] * T_MEASURE
    e_tx = params["p_tx"] * T_TX

    energy_ready_count = np.zeros(n, dtype=int)
    delivered_count = np.zeros(n, dtype=int)

    energy_fail_count = np.zeros(n, dtype=int)
    channel_fail_count = np.zeros(n, dtype=int)

    t_first_success = np.full(n, np.nan)

    for t in range(STEPS):

        # 1. Поступление энергии за шаг моделирования
        e_store = np.minimum(params["e_cap"], e_store + p_h * 60.0)

        # 2. Расход энергии в спящем режиме
        enough_sleep = e_store >= E_SLEEP
        e_store = np.where(enough_sleep, e_store - E_SLEEP, 0.0)

        # 3. Измерение
        if TIME_MIN[t] % MEASURE_INTERVAL_MIN == 0:
            enough_measure = e_store >= e_measure
            e_store = np.where(enough_measure, e_store - e_measure, e_store)

        # 4. Попытка передачи
        if TIME_MIN[t] % TX_INTERVAL_MIN == 0:

            # Энергетическая готовность
            b_en = e_store >= e_tx

            energy_ready_count += b_en
            energy_fail_count += ~b_en

            # Радиоканал проверяется только если энергии хватает
            if np.any(b_en):
                p_rx_bs = received_backscatter_power(
                    distance_m=distance_m,
                    env=env,
                    rng=rng
                )

                b_ch = p_rx_bs >= CHANNEL_PARAMS["P_sens"]
                b_succ = b_en & b_ch

                delivered_count += b_succ
                channel_fail_count += b_en & (~b_ch)

                # энергия тратится на попытку передачи, если устройство было готово
                e_store = np.where(b_en, e_store - e_tx, e_store)

                first_success_mask = np.isnan(t_first_success) & b_succ
                t_first_success[first_success_mask] = t / 60.0

    total_attempts = MAX_TX * n

    return {
        "env": env_name,
        "distance_m": distance_m,
        "mean_energy_ready": energy_ready_count.mean(),
        "mean_delivered": delivered_count.mean(),
        "delivery_probability": delivered_count.sum() / total_attempts,
        "energy_fail_share": energy_fail_count.sum() / total_attempts,
        "channel_fail_share": channel_fail_count.sum() / total_attempts,
        "mean_first_success_h": np.nanmean(t_first_success)
    }


# Таблица результатов

def print_integrated_table(distance_m=3, n=N_MC, seed=SEED):
    print(f"\nИНТЕГРИРОВАННОЕ МОДЕЛИРОВАНИЕ RF + BACKSCATTER")
    print(f"N = {n}, расстояние = {distance_m} м")
    print("-" * 115)

    header = (
        f"{'Среда':<22} "
        f"{'Средн. энерг. готов.':>20} "
        f"{'Средн. доставлено':>18} "
        f"{'P доставки':>12} "
        f"{'Отказ энергия':>14} "
        f"{'Отказ канал':>12}"
    )

    print(header)
    print("-" * 115)

    rows = []

    for env_name in ENVIRONMENTS:
        res = simulate_integrated_rf_backscatter(
            env_name=env_name,
            distance_m=distance_m,
            n=n,
            seed=seed
        )

        rows.append(res)

        print(
            f"{env_name:<22} "
            f"{res['mean_energy_ready']:20.2f} "
            f"{res['mean_delivered']:18.2f} "
            f"{res['delivery_probability']:12.3f} "
            f"{res['energy_fail_share']:14.3f} "
            f"{res['channel_fail_share']:12.3f}"
        )

    return rows


# График вероятности доставки от расстояния

def plot_delivery_probability_by_distance(
    distances=np.arange(1, 11, 1),
    n=N_MC,
    seed=SEED
):
    plt.figure(figsize=(9, 5))

    for env_name in ENVIRONMENTS:
        probabilities = []

        for distance in distances:
            res = simulate_integrated_rf_backscatter(
                env_name=env_name,
                distance_m=distance,
                n=n,
                seed=seed
            )

            probabilities.append(res["delivery_probability"])

        plt.plot(distances, probabilities, marker="o", label=env_name)

    plt.xlabel("Расстояние, м")
    plt.ylabel("Вероятность успешной доставки")
    plt.title("Вероятность успешной доставки при RF-сборе энергии")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("integrated_rf_delivery_probability.png", dpi=300)
    plt.show()


# Диаграмма причин отказов

def plot_failure_reasons(distance_m=3, n=N_MC, seed=SEED):
    rows = print_integrated_table(distance_m=distance_m, n=n, seed=seed)

    labels = [row["env"] for row in rows]
    energy_fail = [row["energy_fail_share"] for row in rows]
    channel_fail = [row["channel_fail_share"] for row in rows]

    x = np.arange(len(labels))

    plt.figure(figsize=(9, 5))
    plt.bar(x, energy_fail, label="Отказ по энергии")
    plt.bar(x, channel_fail, bottom=energy_fail, label="Отказ по радиоканалу")

    plt.xticks(x, labels)
    plt.ylabel("Доля отказов")
    plt.title(f"Причины неуспешных передач при RF-сборе энергии, d={distance_m} м")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig("integrated_rf_failure_reasons.png", dpi=300)
    plt.show()


# Запуск моделирования

if __name__ == "__main__":
    DISTANCE_M = 3

    print_integrated_table(
        distance_m=DISTANCE_M,
        n=N_MC,
        seed=SEED
    )

    plot_delivery_probability_by_distance(
        distances=np.arange(1, 11, 1),
        n=N_MC,
        seed=SEED
    )

    plot_failure_reasons(
        distance_m=DISTANCE_M,
        n=N_MC,
        seed=SEED
    )