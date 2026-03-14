import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

# ── Colour palette (matched from efficiency.jpg / Tableau-10) ──
C = {
    'VideoLLM-Online': '#4E79A7',
    'FVStream':        '#F28E2B',
    'StreamForest':    '#B07AA1',
    'MMDuet-2':        '#E15759',
    'Dispider':        '#76B7B2',
    'StreamAgent':     '#EDC948',
    'Em-Garde (Ours)': '#59A14F',
}

# ── Helper ──
def grouped_bar(ax, metrics, models, data, colors, bar_width=0.12):
    """
    metrics : list of str          – group labels on x-axis
    models  : list of str          – legend entries
    data    : dict[model] -> list   – one value per metric (use None for missing)
    colors  : dict[model] -> str
    """
    n_metrics = len(metrics)
    n_models = len(models)
    x = np.arange(n_metrics)
    total_width = bar_width * n_models
    offsets = np.linspace(-total_width / 2 + bar_width / 2,
                           total_width / 2 - bar_width / 2,
                           n_models)

    ylim_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 100
    label_pad = ylim_top * 0.015

    labeled = set()  # track which models already have a legend entry
    for i, model in enumerate(models):
        vals = data[model]
        for j, v in enumerate(vals):
            if v is None:
                continue
            show_label = model not in labeled
            ax.bar(x[j] + offsets[i], v, bar_width,
                   color=colors[model], edgecolor='white', linewidth=0.5,
                   label=model if show_label else None, zorder=3)
            if show_label:
                labeled.add(model)
            ax.text(x[j] + offsets[i], v + label_pad, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=6.5, color='#333',
                    rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', fontsize=7.5, frameon=True, framealpha=0.9,
              edgecolor='#ddd', ncol=2)


# ═══════════════════════════════════════════════════════
#  Chart 1 – Proactive Response Results
# ═══════════════════════════════════════════════════════
proactive_metrics = [
    'OVO-Bench\nAvg F1',
    'StreamingBench\nPO Accuracy',
    'ProactiveVideoQA\nAvg PAUC',
]

proactive_models = [
    'VideoLLM-Online', 'FVStream', 'StreamForest', 'Dispider',
    'MMDuet-2', 'Em-Garde (Ours)',
]

# ProactiveVideoQA avg PAUC (average of WEB / EGO / VAD)
pvqa_vllm_online = round((25.9 + 25.0 + 25.0) / 3, 1)   # 25.3
pvqa_mmduet2     = round((53.3 + 33.6 + 28.9) / 3, 1)    # 38.6
pvqa_emgarde     = round((44.3 + 52.3 + 27.4) / 3, 1)    # 41.3

proactive_data = {
    'VideoLLM-Online': [6.93,  4.0,  pvqa_vllm_online],
    'FVStream':        [4.77,  2.0,  None],
    'StreamForest':    [13.95, None, None],
    'Dispider':        [None,  25.3, None],
    'MMDuet-2':        [20.51, 34.6, pvqa_mmduet2],
    'Em-Garde (Ours)': [30.99, 38.0, pvqa_emgarde],
}

ymax1 = max(v for vals in proactive_data.values() for v in vals if v is not None)
fig1, ax1 = plt.subplots(figsize=(9.5, 4.5))
ax1.set_ylim(0, ymax1 * 1.22)
grouped_bar(ax1, proactive_metrics, proactive_models, proactive_data, C, bar_width=0.11)
ax1.set_title('Proactive Response Results', fontsize=13, fontweight='bold', pad=12)
ax1.set_ylim(0, ymax1 * 1.22)
fig1.savefig('/home/zhengyikai/Streaming_video/Em_Garde_Streaming_Release/Em-Garde/docs/assets/proactive_results.jpg')
plt.close(fig1)
print('Saved proactive_results.jpg')

# ═══════════════════════════════════════════════════════
#  Chart 2 – Online VideoQA Results
# ═══════════════════════════════════════════════════════
online_metrics = [
    'StreamingBench\nReal-time VU',
    'OVO-Bench\nReal-time VP',
    'OVO-Bench\nBackward Tracing',
]

online_models = [
    'VideoLLM-Online', 'FVStream', 'Dispider', 'StreamAgent',
    'StreamForest', 'Em-Garde (Ours)',
]

online_data = {
    'VideoLLM-Online': [36.0, 20.8, 17.7],
    'FVStream':        [23.2, 29.9, 25.4],
    'Dispider':        [67.6, 54.6, 36.1],
    'StreamAgent':     [74.3, 61.3, 46.2],
    'StreamForest':    [77.3, 61.2, 52.0],
    'Em-Garde (Ours)': [76.7, 63.0, 52.2],
}

ymax2 = max(v for vals in online_data.values() for v in vals if v is not None)
fig2, ax2 = plt.subplots(figsize=(9.5, 4.5))
ax2.set_ylim(0, ymax2 * 1.32)
grouped_bar(ax2, online_metrics, online_models, online_data, C, bar_width=0.11)
ax2.set_title('Online VideoQA Results', fontsize=13, fontweight='bold', pad=12)
ax2.set_ylim(0, ymax2 * 1.32)
fig2.savefig('/home/zhengyikai/Streaming_video/Em_Garde_Streaming_Release/Em-Garde/docs/assets/online_vqa_results.jpg')
plt.close(fig2)
print('Saved online_vqa_results.jpg')
