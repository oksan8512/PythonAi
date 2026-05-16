import matplotlib.pyplot as plt
import numpy as np

# 1 

months = ['Січень', 'Лютий', 'Березень', 'Квітень', 'Травень', 'Червень',
          'Липень', 'Серпень', 'Вересень', 'Жовтень', 'Листопад', 'Грудень']
plan = [120, 135, 128, 150, 160, 175, 168, 180, 172, 190, 185, 200]
fact = [110, 140, 122, 155, 158, 170, 175, 178, 168, 185, 190, 195]

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(months, plan, marker='o', label='План', color='steelblue', linewidth=2)
ax1.plot(months, fact, marker='s', linestyle='--', label='Факт',
         color='seagreen', linewidth=2)
ax1.set_title('Планові та фактичні продажі за місяцями')
ax1.set_xlabel('Місяць')
ax1.set_ylabel('Продажі (тис. грн)')
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# 2 

np.random.seed(42)
ages = np.random.randint(18, 80, size=100)
mean_age = ages.mean()

fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.hist(ages, bins=10, color='steelblue', edgecolor='white', alpha=0.8)
ax2.axvline(mean_age, color='crimson', linewidth=2,
            linestyle='--', label=f'Середнє: {mean_age:.1f} р.')
ax2.set_title('Розподіл вікових груп (100 осіб)')
ax2.set_xlabel('Вік (роки)')
ax2.set_ylabel('Кількість осіб')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# 3

np.random.seed(7)
group_a = np.random.normal(80, 8, 30).clip(50, 100)
group_b = np.random.normal(65, 10, 30).clip(40, 95)
group_c = np.random.normal(88, 5, 30).clip(70, 100)
data_groups = [group_a, group_b, group_c]

fig3, ax3 = plt.subplots(figsize=(8, 5))
bp = ax3.boxplot(data_groups, patch_artist=True, notch=False,
                 medianprops=dict(color='white', linewidth=2))
colors_box = ['#5DCAA5', '#378ADD', '#D85A30']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)
ax3.set_xticks([1, 2, 3])
ax3.set_xticklabels(['Група A', 'Група B', 'Група C'])
ax3.set_title('Розподіл оцінок за групами студентів')
ax3.set_xlabel('Група')
ax3.set_ylabel('Оцінка (балів)')
ax3.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# 4

dates = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Нд']
temperature = [18, 21, 23, 20, 17, 15, 19]
humidity    = [55, 60, 65, 70, 68, 72, 58]

fig4, ax4 = plt.subplots(figsize=(9, 5))
ax4.plot(dates, temperature, marker='o', color='tomato',
         linewidth=2, label='Температура (°C)')
ax4.plot(dates, humidity, marker='s', linestyle='--', color='steelblue',
         linewidth=2, label='Вологість (%)')
ax4.set_title('Температура та вологість протягом тижня')
ax4.set_xlabel('День тижня')
ax4.set_ylabel('Значення')
ax4.legend()
ax4.grid(alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 5

hours = list(range(24))
server_load = [12, 8, 6, 5, 7, 10, 15, 30, 55, 72, 80, 78,
               75, 70, 65, 68, 74, 78, 60, 45, 35, 28, 20, 15]

fig5, ax5 = plt.subplots(figsize=(11, 5))
ax5.plot(hours, server_load, color='#3B6D11', linewidth=2, marker='.')
ax5.fill_between(hours, server_load, alpha=0.25, color='#639922')
ax5.set_title('Навантаження сервера протягом доби')
ax5.set_xlabel('Година')
ax5.set_ylabel('Навантаження (%)')
ax5.set_xticks(hours)
ax5.set_xticklabels([f'{h}:00' for h in hours], rotation=45, fontsize=8)
ax5.set_ylim(0, 100)
ax5.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#  6 

weeks = [f'Т{i}' for i in range(1, 9)]
conversion   = [3.2, 3.5, 3.1, 3.8, 4.0, 3.7, 4.2, 4.5]
retention    = [62,  65,  63,  67,  70,  68,  72,  75 ]
avg_check    = [420, 430, 415, 445, 460, 450, 470, 490]
orders_count = [320, 350, 338, 370, 390, 365, 405, 420]

metrics = [
    ('Конверсія (%)',         conversion,   'steelblue'),
    ('Утримання (%)',         retention,    'seagreen'),
    ('Середній чек (грн)',    avg_check,    'darkorange'),
    ('Кількість замовлень',   orders_count, 'mediumpurple'),
]

fig6, axes = plt.subplots(2, 2, figsize=(12, 7))
fig6.suptitle('Метрики продукту за 8 тижнів', fontsize=14, fontweight='bold')
for ax, (title, data, color) in zip(axes.flat, metrics):
    ax.plot(weeks, data, marker='o', color=color, linewidth=2)
    ax.fill_between(weeks, data, alpha=0.15, color=color)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Тиждень', fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()