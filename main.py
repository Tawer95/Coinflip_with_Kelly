import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# === Параметры ===
num_games = 1000                # количество бросков
num_simulations = 25000         # количество симуляций для графика
starting_stack = 1000           # стартовый банкролл

prob_win = 0.5                  # вероятность победы
gain_factor = 2                 # остаток от ставки после победы
lose_factor = 0.4               # остаток от ставки после проигрыша

Kelly_criterion = True          # расчитывать критерий Келли
manual_fraction = None          # кастомный метод для изменения ставки вопреки оптимальном по критерию Келли, если None, то он сам будет высчитывать оптимальный

# === Комиссия ===
fee_on = True                  # включены ли комиссии площадки
fee_rate = 0.001                # размер комиссии, 0.001 = 0.1% комиссии


def get_kelly_fraction(p, gain, loss, fee=0):
    def neg_kelly(f):
        if f <= 0 or f >= 1/loss:
            return -np.inf
        try:
            return -(p * np.log(1 + gain * f - fee * f) +
                     (1 - p) * np.log(1 - loss * f - fee * f))
        except:
            return -np.inf
    result = minimize_scalar(neg_kelly, bounds=(0, 1/loss-1e-6), method='bounded')
    return result.x

p = prob_win
q = 1 - prob_win
kelly_gain = gain_factor - 1
kelly_loss = 1 - lose_factor

def adjusted_ev(p, g, l, fee, kelly_f):
    return p * (1 + g * kelly_f - fee * kelly_f) + \
           q * (1 - l * kelly_f - fee * kelly_f)

if Kelly_criterion:
    if manual_fraction is not None:
        kelly_f = manual_fraction
        kelly_note = f"Используется ручная ставка: {kelly_f:.4f} от капитала"
    else:
        kelly_f = get_kelly_fraction(p, kelly_gain, kelly_loss, fee_rate if fee_on else 0)
        kelly_note = f"Оптимальная ставка по Келли: {kelly_f:.4f} от капитала"
else:
    kelly_f = 1.0
    kelly_note = f"Используется весь капитал (Келли выключен)"

if fee_on:
    effective_gain = gain_factor - 1 - fee_rate
    effective_loss = 1 - lose_factor + fee_rate
    ev_per_flip = p * (1 + effective_gain * kelly_f) + q * (1 - effective_loss * kelly_f)
    ev_percent = (ev_per_flip - 1) * 100
    fee_text = f"\nКомиссия: {fee_rate*100:.3f}% за бросок (активна)"
else:
    ev_per_flip = p * gain_factor + q * lose_factor
    ev_percent = (ev_per_flip - 1) * 100
    fee_text = f"\nКомиссия не применяется"
kelly_note += f"\nМатематическое ожидание одной ставки: {ev_per_flip:.4f} ({ev_percent:+.2f}%)" + fee_text

# === Симуляция ===
results = np.zeros((num_simulations, num_games + 1))
results[:, 0] = starting_stack

for i in range(num_simulations):
    stack = starting_stack
    for j in range(1, num_games + 1):
        if Kelly_criterion:
            bet = stack * kelly_f
            fee_amt = fee_rate * bet if fee_on else 0
            rest = stack - bet
            if np.random.rand() < prob_win:
                stack = rest + bet * gain_factor - fee_amt
            else:
                stack = rest + bet * lose_factor - fee_amt
        else:
            if fee_on:
                bet = stack
                fee_amt = fee_rate * bet
                if np.random.rand() < prob_win:
                    stack = stack * gain_factor - fee_amt
                else:
                    stack = stack * lose_factor - fee_amt
            else:
                if np.random.rand() < prob_win:
                    stack *= gain_factor
                else:
                    stack *= lose_factor
        results[i, j] = stack

percentiles = [5, 25, 75, 95]
percentile_lines = np.percentile(results, percentiles, axis=0)
mean_line = results.mean(axis=0)
geom_mean = np.exp(np.mean(np.log(np.where(results > 0, results, 1e-300)), axis=0))

theoretical_ev = np.zeros(num_games + 1)
theoretical_ev[0] = starting_stack
for i in range(1, num_games + 1):
    theoretical_ev[i] = theoretical_ev[i-1] * ev_per_flip

# ==== График ====
fig, ax = plt.subplots(figsize=(14, 8))
ax.grid(True, which="both", ls=":", lw=0.8, color="gray", alpha=0.65)
ax.minorticks_on()

# 200 случайных траекторий
for i in np.random.choice(num_simulations, size=min(300, num_simulations), replace=False):
    ax.plot(results[i], color='grey', alpha=0.12, linewidth=1)

colors = ['brown', 'green', 'violet', 'crimson']
labels = [f'{p}% процентиль' for p in percentiles]

for i, p_line in enumerate(percentile_lines):
    ax.plot(p_line, color=colors[i], label=labels[i], linewidth=2 if percentiles[i] in [5, 50, 95] else 1.5)


# Арифметическое среднее (красная пунктирная)
ax.plot(mean_line, color='r', linestyle='--', linewidth=2, label='Арифметическое среднее')

# Геометрическое среднее (синяя пунктирная) + яркая точка на конце
ax.plot(geom_mean, color='b', linestyle='--', linewidth=2, label='Геометрическое среднее')

# Теоретическое ожидание (зелёная пунктирная)
ax.plot(theoretical_ev, 'g--', linewidth=2, label='Математическое ожидание', zorder=4)

# Начальный банкролл — жирная чёрная линия
ax.axhline(starting_stack, color='black', lw=2, label=f'Начальный банкролл ({starting_stack}$)', zorder=3)

ax.set_yscale('log')
ax.set_xlabel("Time (coins flipped)")
ax.set_ylabel("Wealth (log scale)")
ax.set_xlim(0, num_games + 60)
ax.set_title(f"Coin Flipping Game Simulation\n{num_games} flips, {num_simulations} paths\n{kelly_note}")

legend = ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False)
legend.get_frame().set_alpha(1)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_linewidth(1.1)

plt.tight_layout()
plt.show()