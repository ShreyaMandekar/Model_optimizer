# experiments/04_paper_figures.py

import torch
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs('results/figures', exist_ok=True)

# ── Figure 1: profiler op breakdown ─────────────────────────
# shows WHERE the 69ms goes in QAT vs FP32

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# QAT breakdown from your profiler output
qat_ops  = ['matmul\n(aten::mm)', 'FakeQuantize\n(_Generic)', 
             'choose_qparams', 'aten::amin', 'other']
qat_time = [67, 47, 27, 14, 6]
colors   = ['#4878CF', '#E84646', '#E84646', '#E84646', '#888888']

bars = axes[0].bar(qat_ops, qat_time, color=colors, width=0.6,
                   edgecolor='white', linewidth=0.5)
axes[0].set_title('QAT compiled — op breakdown', fontsize=12)
axes[0].set_ylabel('CUDA time (ms)', fontsize=11)
axes[0].set_ylim(0, 80)
for bar, val in zip(bars, qat_time):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val}ms', ha='center', va='bottom', fontsize=10)

useful  = mpatches.Patch(color='#4878CF', label='Useful compute')
wasted  = mpatches.Patch(color='#E84646', label='Quantization overhead')
axes[0].legend(handles=[useful, wasted], fontsize=9)

# FP32 breakdown
fp32_ops  = ['matmul\n(sgemm)', 'flash\nattention', 'layer\nnorm', 'other']
fp32_time = [99, 4, 2, 1]
colors2   = ['#4878CF', '#4878CF', '#4878CF', '#888888']

bars2 = axes[1].bar(fp32_ops, fp32_time, color=colors2, width=0.6,
                    edgecolor='white', linewidth=0.5)
axes[1].set_title('FP32 compiled — op breakdown', fontsize=12)
axes[1].set_ylabel('CUDA time (ms)', fontsize=11)
axes[1].set_ylim(0, 115)
for bar, val in zip(bars2, fp32_time):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val}ms', ha='center', va='bottom', fontsize=10)

plt.suptitle('Figure 1: QAT spends 55% of GPU time on quantization overhead',
             fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig1_profiler_breakdown.pdf',
            bbox_inches='tight', dpi=150)
plt.savefig('results/figures/fig1_profiler_breakdown.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig1")

# ── Figure 2: CV distribution histogram ─────────────────────
# shows that most layers have very low scale variance

cv_values = [
    5.69, 5.69, 5.69, 3.94, 2.81, 1.83,   # layer 0
    1.53, 1.53, 1.53, 2.93, 4.61, 1.65,   # layer 1
    3.06, 3.06, 3.06, 4.40, 2.70, 2.64,   # layer 2
    2.42, 2.42, 2.42, 3.42, 1.89, 3.45,   # layer 3
    2.97, 2.97, 2.97, 3.14, 2.48, 5.70,   # layer 4
    3.72, 3.72, 3.72, 4.76, 3.87, 4.25,   # layer 5
    6.23, 4.18                              # pre_classifier, classifier
]

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(cv_values, bins=20, color='#4878CF', edgecolor='white',
        linewidth=0.5, alpha=0.85)
ax.axvline(5.0, color='#E84646', linestyle='--', linewidth=1.5,
           label='CV = 5% threshold (STATIC OK)')
ax.axvline(15.0, color='#E87010', linestyle='--', linewidth=1.5,
           label='CV = 15% threshold (DYNAMIC)')

static_count = sum(1 for v in cv_values if v < 5)
ax.text(2.5, 9.5, f'{static_count}/38 layers\nCV < 5%',
        fontsize=11, color='#4878CF', ha='center')

ax.set_xlabel('Coefficient of variation (%)', fontsize=11)
ax.set_ylabel('Number of layers', fontsize=11)
ax.set_title('Figure 2: Activation scale CV across 38 QAT layers\n'
             '(measured on 200 SST-2 validation sentences)',
             fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(0, 20)
plt.tight_layout()
plt.savefig('results/figures/fig2_cv_distribution.pdf',
            bbox_inches='tight', dpi=150)
plt.savefig('results/figures/fig2_cv_distribution.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig2")

# ── Figure 3: threshold ablation Pareto curve ────────────────
with open('results/profiles/threshold_ablation.json') as f:
    ablation = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

thresholds = [r['threshold'] for r in ablation]
accuracies = [r['accuracy']  for r in ablation]
latencies  = [r['latency']   for r in ablation]
converted  = [r['converted'] for r in ablation]

# Pareto curve: latency vs accuracy
sc = ax1.scatter(latencies, accuracies, c=thresholds,
                 cmap='viridis', s=100, zorder=5)
for r in ablation:
    ax1.annotate(f"cv<{r['threshold']}%",
                 (r['latency'], r['accuracy']),
                 textcoords='offset points', xytext=(6, 3), fontsize=8)
ax1.axhline(87.38, color='steelblue', linestyle='--',
            linewidth=1, label='QAT dynamic (87.38%)')
ax1.axvline(12.0, color='green', linestyle='--',
            linewidth=1, label='FP32 latency (12ms)')
ax1.set_xlabel('Latency (ms)', fontsize=11)
ax1.set_ylabel('SST-2 accuracy (%)', fontsize=11)
ax1.set_title('Accuracy vs latency (threshold sweep)', fontsize=11)
ax1.legend(fontsize=9)
ax1.set_ylim(83, 89)
ax1.grid(True, alpha=0.3)
plt.colorbar(sc, ax=ax1, label='CV threshold (%)')

# layers converted vs accuracy
ax2.plot(converted, accuracies, 'o-', color='coral',
         linewidth=1.5, markersize=8)
for i, r in enumerate(ablation):
    ax2.annotate(f"cv<{r['threshold']}%",
                 (r['converted'], r['accuracy']),
                 textcoords='offset points', xytext=(4, 4), fontsize=8)
ax2.axhline(87.38, color='steelblue', linestyle='--',
            linewidth=1, label='QAT dynamic baseline')
ax2.set_xlabel('Layers converted to static', fontsize=11)
ax2.set_ylabel('SST-2 accuracy (%)', fontsize=11)
ax2.set_title('Non-monotonic accuracy vs layers converted', fontsize=11)
ax2.legend(fontsize=9)
ax2.set_ylim(83, 89)
ax2.grid(True, alpha=0.3)

plt.suptitle('Figure 3: CV threshold ablation', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig('results/figures/fig3_threshold_ablation.pdf',
            bbox_inches='tight', dpi=150)
plt.savefig('results/figures/fig3_threshold_ablation.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig3")

# ── Figure 4: main results bar chart ────────────────────────
# fill in lat_combined and acc_combined from Step 1 output
# placeholder values — replace with your actual numbers
lat_combined = 12.5   # REPLACE with actual
acc_combined = 87.96  # REPLACE with actual

models    = ['FP32\nbaseline', 'QAT\ndynamic',
             'Phase 2\npost-hoc', 'Phase 3\ntraining-aware',
             'Phase 3\n+ cleanup']
latencies_bar = [12.0, 34.8, 12.0, 19.3, lat_combined]
accuracies_bar = [None, 87.38, 86.69, 87.96, acc_combined]
bar_colors     = ['#888888', '#E84646', '#4878CF', '#E87010', '#2CA02C']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

bars = ax1.bar(models, latencies_bar, color=bar_colors,
               edgecolor='white', linewidth=0.5, width=0.6)
ax1.axhline(12.0, color='black', linestyle='--',
            linewidth=1, alpha=0.4, label='FP32 reference (12ms)')
ax1.set_ylabel('Latency (ms)', fontsize=11)
ax1.set_title('Figure 4: Latency and accuracy across all models', fontsize=12)
ax1.legend(fontsize=9)
ax1.set_ylim(0, 42)
for bar, val in zip(bars, latencies_bar):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)

acc_vals   = [v for v in accuracies_bar if v is not None]
acc_models = [m for m, v in zip(models, accuracies_bar) if v is not None]
acc_colors = [c for c, v in zip(bar_colors, accuracies_bar) if v is not None]

bars2 = ax2.bar(acc_models, acc_vals, color=acc_colors,
                edgecolor='white', linewidth=0.5, width=0.6)
ax2.axhline(87.38, color='#E84646', linestyle='--',
            linewidth=1, alpha=0.6, label='QAT dynamic baseline (87.38%)')
ax2.set_ylabel('SST-2 accuracy (%)', fontsize=11)
ax2.set_ylim(84, 89.5)
ax2.legend(fontsize=9)
for bar, val in zip(bars2, acc_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/figures/fig4_main_results.pdf',
            bbox_inches='tight', dpi=150)
plt.savefig('results/figures/fig4_main_results.png',
            bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig4")

print("\nAll figures saved to results/figures/")
print("Update lat_combined and acc_combined in fig4 with actual numbers from Step 1")