import numpy as np

# Общие параметры моделирования
DT_MIN = 1
HOURS = 48
STEPS = HOURS * 60 // DT_MIN
TIME_MIN = np.arange(STEPS)
TIME_H = TIME_MIN / 60.0
HOUR_OF_DAY = TIME_H % 24.0


# Параметры Ambient IoT-устройства
VDD = 3.0
P_SLEEP = (0.94e-6 + 0.30e-6) * VDD     # спящий режим
P_ACTIVE = 3.4e-3 * VDD                 # активный режим
P_TX = 7.0e-3 * VDD                     # передача данных
T_MEASURE = 0.5                         # длительность измерения
T_TX = 0.1                              # длительность передачи
MEASURE_INTERVAL_MIN = 5                # измерение каждые 5 минут
TX_INTERVAL_MIN = 15                    # передача каждые 15 минут
E_SLEEP = P_SLEEP * 60.0
E_MEASURE = P_ACTIVE * T_MEASURE
E_TX = P_TX * T_TX
E_CAP_DEFAULT = 1.8                     # полезная энергия накопителя

P_LOAD_AVG = (E_SLEEP / 60.0 + E_MEASURE / (MEASURE_INTERVAL_MIN * 60.0) + E_TX / (TX_INTERVAL_MIN * 60.0))


# Параметры источников сбора энергии
ETA_SOLAR = 0.18
AREA_SOLAR = 1e-4                      
PT_RF = 1.0
GT = 1.0
GR = 1.0
F_RF = 2.4e9
C = 3e8
ETA_RF = 0.5
LAMBDA = C / F_RF
K_TEG = 2e-6
K_PIEZO = 4e-4

SOURCES = ["Солнечный","Радиочастотный","Термоэлектрический","Пьезоэлектрический"]


# Профили сред эксплуатации

def indoor_irradiance(hour):
    x = np.full_like(hour, 0.05, dtype=float)
    x[(hour >= 7) & (hour < 8)] = 0.6
    x[(hour >= 8) & (hour < 18)] = 1.5
    x[(hour >= 18) & (hour < 22)] = 0.6
    return x

def outdoor_irradiance(hour):
    x = np.sin(np.pi * (hour - 6) / 12.0)
    return np.where((hour >= 6) & (hour <= 18),1000.0 * np.maximum(0.0, x),0.0)

def industrial_irradiance(hour):
    x = np.full_like(hour, 0.1, dtype=float)
    x[(hour >= 6) & (hour < 22)] = 2.1
    return x

def indoor_dT(hour):
    x = np.full_like(hour, 1.0, dtype=float)
    x[(hour >= 6) & (hour < 22)] = 2.0
    return x

def outdoor_dT(hour):
    x = np.full_like(hour, 1.0, dtype=float)
    x[(hour >= 9) & (hour < 17)] = 3.0
    return x

def industrial_dT(hour):
    x = np.full_like(hour, 8.0, dtype=float)
    x[(hour >= 6) & (hour < 22)] = 15.0
    return x

def indoor_vibration(hour):
    x = np.full_like(hour, 0.01, dtype=float)
    x[(hour >= 8) & (hour < 20)] = 0.05
    return x

def outdoor_vibration(hour):
    x = np.full_like(hour, 0.02, dtype=float)
    x[(hour >= 8) & (hour < 20)] = 0.08
    return x

def industrial_vibration(hour):
    x = np.full_like(hour, 0.15, dtype=float)
    x[(hour >= 6) & (hour < 22)] = 0.50
    return x

ENVIRONMENTS = {
    "В помещении": {
        "irradiance": indoor_irradiance(HOUR_OF_DAY),
        "dT": indoor_dT(HOUR_OF_DAY),
        "vibration": indoor_vibration(HOUR_OF_DAY),
        "rf_distance": 4.0,
    },
    "Уличная среда": {
        "irradiance": outdoor_irradiance(HOUR_OF_DAY),
        "dT": outdoor_dT(HOUR_OF_DAY),
        "vibration": outdoor_vibration(HOUR_OF_DAY),
        "rf_distance": 12.0,
    },
    "Промышленная среда": {
        "irradiance": industrial_irradiance(HOUR_OF_DAY),
        "dT": industrial_dT(HOUR_OF_DAY),
        "vibration": industrial_vibration(HOUR_OF_DAY),
        "rf_distance": 3.0,
    },
}

# Генерация случайных параметров Монте-Карло

def sample_mc_params(env_name, n, rng):
    base_d = ENVIRONMENTS[env_name]["rf_distance"]
    return {
        "area": np.clip(rng.normal(AREA_SOLAR, AREA_SOLAR * 0.30, n),0.4e-4,2.5e-4),
        "d_rf": np.clip(base_d * rng.lognormal(0.0, 0.35, n),0.5,25.0),
        "k_teg": np.clip(K_TEG * rng.lognormal(0.0, 0.30, n),0.3 * K_TEG,3.0 * K_TEG),
        "k_piezo": np.clip(K_PIEZO * rng.lognormal(0.0, 0.35, n),0.2 * K_PIEZO,4.0 * K_PIEZO),
        "e_init": rng.uniform(0.2 * E_CAP_DEFAULT,E_CAP_DEFAULT,n),
        "e_cap": np.clip(rng.normal(E_CAP_DEFAULT, 0.35*E_CAP_DEFAULT,n), 0.5*E_CAP_DEFAULT, 3.0*E_CAP_DEFAULT),
        "p_tx": np.clip(P_TX * rng.lognormal(0.0, 0.25, n),0.4 * P_TX,3.0 * P_TX),
        "p_active": np.clip(P_ACTIVE * rng.lognormal(0.0, 0.15, n),0.6 * P_ACTIVE,2.0 * P_ACTIVE),
    }

# Монте-Карло моделирование

def simulate_monte_carlo(env_name, source_name, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    params = sample_mc_params(env_name, n, rng)
    env = ENVIRONMENTS[env_name]
    irr = env["irradiance"][:, None]
    dT = env["dT"][:, None]
    vib = env["vibration"][:, None]
    rf_term = ETA_RF * PT_RF * GT * GR * (LAMBDA / (4.0 * np.pi * params["d_rf"][None, :])) ** 2

    if source_name == "Солнечный":
        p_h = ETA_SOLAR * params["area"][None, :] * irr
    elif source_name == "Радиочастотный":
        p_h = np.ones((STEPS, n)) * rf_term
    elif source_name == "Термоэлектрический":
        p_h = params["k_teg"][None, :] * dT ** 2
    elif source_name == "Пьезоэлектрический":
        p_h = params["k_piezo"][None, :] * vib ** 2
    else:
        raise ValueError("Ошибка")

    e_store = params["e_init"].copy()
    e_measure = params["p_active"] * T_MEASURE
    e_tx = params["p_tx"] * T_TX
    success_tx = np.zeros(n, dtype=int)
    failed_tx = np.zeros(n, dtype=int)
    t_depletion = np.full(n, np.nan)

    for t in range(STEPS):
        # Поступление энергии за шаг моделирования
        e_store = np.minimum(params["e_cap"], e_store + p_h[t] * 60.0)

        # Потребление в спящем режиме
        enough_sleep = e_store >= E_SLEEP
        e_store = np.where(enough_sleep, e_store - E_SLEEP, 0.0)

        hit_zero = np.isnan(t_depletion) & (e_store <= 1e-12)
        t_depletion[hit_zero] = t / 60.0

        # Измерение
        if TIME_MIN[t] % MEASURE_INTERVAL_MIN == 0:
            enough_measure = e_store >= e_measure
            e_store = np.where(enough_measure, e_store - e_measure, e_store)

        # Передача данных
        if TIME_MIN[t] % TX_INTERVAL_MIN == 0:
            enough_tx = e_store >= e_tx
            success_tx += enough_tx
            failed_tx += ~enough_tx
            e_store = np.where(enough_tx, e_store - e_tx, e_store)

        hit_zero = np.isnan(t_depletion) & (e_store <= 1e-12)
        t_depletion[hit_zero] = t / 60.0

    autonomous = (failed_tx == 0) & np.isnan(t_depletion)
    t_depletion = np.where(np.isnan(t_depletion), HOURS, t_depletion)

    return params, {
        "success_tx": success_tx,
        "failed_tx": failed_tx,
        "autonomous": autonomous,
        "t_depletion_h": t_depletion,
        "e_final": e_store.copy(),
    }

# Таблица результатов

def print_mc_table(n=1000, seed=42):
    print(f"\nMONTE CARLO РЕЗУЛЬТАТЫ, N={n}")
    print("-" * 90)

    header = (
        f"{'Среда':<22} "
        f"{'Источник':<20} "
        f"{'Средн. успех':>14} "
        f"{'Средн. t разр.':>16} "
        f"{'P(авт.)':>10}"
    )
    print(header)
    print("-" * 90)

    max_tx = HOURS * 60 // TX_INTERVAL_MIN

    for env_name in ENVIRONMENTS:
        for src in SOURCES:
            _, out = simulate_monte_carlo(env_name, src, n=n, seed=seed)
            print(
                f"{env_name:<22} "
                f"{src:<20} "
                f"{out['success_tx'].mean():14.2f} "
                f"{out['t_depletion_h'].mean():16.2f} "
                f"{out['autonomous'].mean():10.3f}"
            )
    print(f"\nМаксимум передач за {HOURS} часов: {max_tx}")


# Данные для графиков

def collect_plot_data(n=1000, seed=42):

    sources_order = SOURCES
    environments_order = list(ENVIRONMENTS.keys())
    autonomy_probability = {}
    successful_tx = {}

    for env_name in environments_order:
        autonomy_probability[env_name] = []
        successful_tx[env_name] = []

        for src in sources_order:
            _, out = simulate_monte_carlo(env_name, src, n=n, seed=seed)
            autonomy_probability[env_name].append(out["autonomous"].mean() * 100)
            successful_tx[env_name].append(out["success_tx"].mean())

    return {
        "sources_order": sources_order,
        "environments_order": environments_order,
        "autonomy_probability": autonomy_probability,
        "successful_tx": successful_tx,
    }


# Запуск моделирования

if __name__ == "__main__":
    N = 1000
    SEED = 42

    print(f"Средняя мощность нагрузки: "f"{P_LOAD_AVG * 1e6:.2f} мкВт")
    print_mc_table(n=N, seed=SEED)
    plot_data = collect_plot_data(n=N, seed=SEED)