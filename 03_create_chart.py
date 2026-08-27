"""
Task 2 - Step 3: Feature importance chart
"""
import pandas as pd
import matplotlib.pyplot as plt

imp = pd.read_csv('outputs/feature_importance.csv').head(10)

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['#075AAA' if i > 2 else '#EF3340' for i in range(len(imp))]
bars = ax.barh(imp['feature'][::-1], imp['importance'][::-1], color=colors[::-1])

ax.set_xlabel('Relative Importance', fontsize=12)
ax.set_title('Top 10 Predictors of Booking Completion', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, val in zip(bars, imp['importance'][::-1]):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.1%}',
            va='center', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/feature_importance_chart.png', dpi=200, bbox_inches='tight')
print("Chart saved to outputs/feature_importance_chart.png")