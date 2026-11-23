import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from .utils import log, load_json, load_json_safe
from .datasets import DATASET_NAMES
from .perturbations import FAMILIES, FAMILY_NAMES, apply_condition
from .engine import SETTINGS


INK = '#4a4a4a'
GREY = '#7a7a7a'
ORANGE = '#c0622f'
PURPLE = '#5e3c99'
BLUE = '#2c6fb0'
TEAL = '#2a8c82'
LIGHT = '#a9a9a9'
METHOD_STYLES = {'FRN': (BLUE, 'v', '-'), 'Bi-FRN': (TEAL, '^', '-'), 'A2': (GREY, 's', '-'), 'C2-max': (PURPLE, 'D', '-'), 'A3': (ORANGE, 'o', '-')}
FIG_NAMES = {'FRN': 'FRN', 'Bi-FRN': 'Bi-FRN', 'A2': 'A2', 'C2-max': 'C2 (shared)', 'A3': 'CaPT-MR'}
PROBE_STYLES = {'A2': GREY, 'A3': ORANGE, 'C2-max': PURPLE, 'C2-mean': BLUE, 'C2-weighted': TEAL, 'Anchored': '#9c6b3e'}
MAIN_NAMES = {'ProtoNet': 'ProtoNet', 'DN4': 'DN4', 'DMN4': 'DMN4', 'DeepEMD': 'DeepEMD', 'FRN': 'FRN', 'Bi-FRN': 'Bi-FRN', 'BSNet': 'BSNet',
              'TDM': 'TDM', 'SFMR': 'SFMR', 'A2': 'Partial OT, no gate (A2)', 'C2-max': 'Shared rival gate (C2-max)', 'A3': 'CaPT-MR (A3)'}
CONFIRM_NAMES = {'A0': 'Dense softmax mutual reconstruction (no OT)', 'A1': 'Balanced OT, rho=1, no gate', 'A2': 'Partial OT, rho=0.6, no gate',
                 'A3': 'A2 + rival-conditioned gate (CaPT-MR)', 'C1': 'Candidate-only gate', 'C2-max': 'Shared gate from max_c m(i,c)',
                 'C3': 'Top-rho rows by m, capacity-free softmax', 'C4': 'A3 gate spatially permuted'}
STYLE = {
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 11, 'axes.labelsize': 11.5, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5, 'legend.fontsize': 10,
    'text.color': INK, 'axes.labelcolor': INK, 'axes.edgecolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'axes.linewidth': 1.2, 'xtick.major.width': 1.2, 'ytick.major.width': 1.2, 'axes.spines.top': False, 'axes.spines.right': False,
    'grid.color': '#dcdcdc', 'grid.linewidth': 0.8, 'legend.frameon': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
}


def key(*parts):
    return '|'.join(str(p) for p in parts)


def setting_label(dataset, shot):
    return DATASET_NAMES[dataset], f'{shot}-shot'


def finite(*values):
    return all(v is not None and np.isfinite(v) for v in values)


def num(value, digits=1, sign=False):
    if value is None or not np.isfinite(float(value)):
        return '--'
    return f'{float(value):+.{digits}f}' if sign else f'{float(value):.{digits}f}'


def pm(m, digits=1):
    if not m or not finite(m.get('mean')):
        return '--'
    return f'{m["mean"]:.{digits}f} ± {m["half"]:.2f}' if finite(m.get('half')) else f'{m["mean"]:.{digits}f}'


def ci(c, simultaneous=True, digits=2):
    if not c or not finite(c.get('mean')):
        return '--'
    lo, hi = (c.get('slo'), c.get('shi')) if simultaneous else (c.get('lo'), c.get('hi'))
    if not finite(lo, hi):
        return num(c['mean'], 1, True)
    return f'{c["mean"]:+.1f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'


def interval(c, simultaneous=False, digits=2):
    if not c:
        return '--'
    lo, hi = (c.get('slo'), c.get('shi')) if simultaneous else (c.get('lo'), c.get('hi'))
    return f'[{lo:+.{digits}f}, {hi:+.{digits}f}]' if finite(lo, hi) else '--'


def latex_cell(value):
    text = str(value)
    for a, b in (('\\', '\\textbackslash{}'), ('&', '\\&'), ('%', '\\%'), ('_', '\\_'), ('#', '\\#'), ('±', '$\\pm$'), ('rho', '$\\rho$')):
        text = text.replace(a, b)
    return text


def write_table(rows, columns, title, out, name, note=''):
    if not rows:
        return False
    frame = pd.DataFrame(rows, columns=columns)
    with open(out / f'{name}.md', 'w') as f:
        f.write(f'### {title}\n\n')
        f.write('| ' + ' | '.join(columns) + ' |\n')
        f.write('|' + '|'.join(['---'] * len(columns)) + '|\n')
        for row in rows:
            f.write('| ' + ' | '.join(str(v) for v in row) + ' |\n')
        if note:
            f.write(f'\n{note}\n')
    with open(out / f'{name}.tex', 'w') as f:
        f.write('\\begin{table}[t]\n\\centering\n')
        f.write(f'\\caption{{{latex_cell(title)}}}\n\\label{{{name}_reproduced}}\n\\footnotesize\n')
        f.write('\\begin{tabular}{@{}' + 'l' * len(columns) + '@{}}\n\\toprule\n')
        f.write(' & '.join(latex_cell(c) for c in columns) + ' \\\\\n\\midrule\n')
        for row in rows:
            f.write(' & '.join(latex_cell(v) for v in row) + ' \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
        if note:
            f.write('\\vspace{2pt}\n\\begin{minipage}{\\linewidth}\\raggedright\\footnotesize ' + latex_cell(note) + '\\end{minipage}\n')
        f.write('\\end{table}\n')
    frame.to_csv(out / f'{name}.csv', index=False)
    return True


def save(fig, out, name):
    fig.savefig(out / f'{name}.pdf')
    fig.savefig(out / f'{name}.png', dpi=300)
    plt.close(fig)


def resource(results, backbone, shot, method, field='median'):
    entry = results.get('resources', {}).get('latency', {}).get(key(backbone, shot, method))
    return entry.get(field) if entry else None


def table_main(results, out):
    rows = []
    for method in MAIN_NAMES:
        cells = [pm(results.get('main', {}).get(key(method, d, k))) for d, k in SETTINGS]
        if any(c != '--' for c in cells):
            rows.append([MAIN_NAMES[method]] + cells)
    columns = ['Method'] + [f'{DATASET_NAMES[d]} {k}-shot' for d, k in SETTINGS]
    write_table(rows, columns, 'Five-way accuracy (%) with marginal 95% Student-t half-widths over training seeds (Conv-64, raw 84x84 inputs)', out,
                'tab_main', note='Baselines are common-protocol adaptations; 2,000 fixed episodes are averaged within each seed.')
    rows = []
    for method in ('FRN', 'Bi-FRN', 'TDM', 'SFMR', 'A2', 'C2-max', 'C2-mean', 'C2-weighted', 'A3'):
        cells = [num(resource(results, 'conv64', k, method), 1) for k in (1, 5)] + [num(resource(results, 'conv64', k, method, 'peak_mb'), 0) for k in (1, 5)]
        if any(c != '--' for c in cells):
            rows.append([method] + cells)
    write_table(rows, ['Method', 'Forward ms 1-shot', 'Forward ms 5-shot', 'Peak MB 1-shot', 'Peak MB 5-shot'],
                'Model-forward latency (ms per 75-query episode) and peak allocated memory (decimal MB)', out, 'tab_main_resources')


def table_components(results, out):
    rows = []
    for d, k in SETTINGS:
        c = results.get('components', {}).get(key(d, k), {})
        if any(c.get(n) for n in ('c2max_a2', 'a3_best_shared', 'mutual_query')):
            rows.append([*setting_label(d, k), ci(c.get('c2max_a2')), ci(c.get('a3_best_shared')), ci(c.get('mutual_query'))])
    write_table(rows, ['Dataset', 'Setting', 'C2-max-A2 [CI]', 'A3-best shared [CI]', 'Mutual-query [CI]'],
                'Paired accuracy differences (pp) with 95% simultaneous intervals', out, 'tab_components',
                note='C2-max-A2: six-setting family (F=6); A3-best shared: minima over the joint 18-contrast family; mutual-query: 12-contrast family.')


def table_shared(results, out):
    rows, contrast_rows, cost_rows = [], [], []
    for d, k in SETTINGS:
        cells = [pm(results.get('main', {}).get(key(m, d, k))).split(' ±')[0] for m in ('C2-max', 'C2-mean', 'C2-weighted', 'A3')]
        if any(c != '--' for c in cells):
            rows.append([*setting_label(d, k)] + cells)
        shared = results.get('shared', {}).get(key(d, k), {})
        if any(shared.values()):
            contrast_rows.append([*setting_label(d, k)] + [ci(shared.get(r)) for r in ('C2-max', 'C2-mean', 'C2-weighted')])
    write_table(rows, ['Dataset', 'Setting', 'C2-max', 'C2-mean', 'C2-weighted', 'A3'], 'Shared versus candidate-specific allocation: accuracy (%)',
                out, 'tab_shared_variants')
    write_table(contrast_rows, ['Dataset', 'Setting', 'A3-max [CI]', 'A3-mean [CI]', 'A3-weighted [CI]'],
                'A3 minus each shared rule (pp), joint 95% simultaneous intervals (F=18)', out, 'tab_shared_variants_contrasts')
    for k in (1, 5):
        for rule in ('C2-max', 'C2-mean', 'C2-weighted'):
            a, b = resource(results, 'conv64', k, 'A3'), resource(results, 'conv64', k, rule)
            ma, mb = resource(results, 'conv64', k, 'A3', 'peak_mb'), resource(results, 'conv64', k, rule, 'peak_mb')
            if finite(a, b):
                cost_rows.append([f'{k}-shot', rule, f'{a:.1f}/{b:.1f}', num(100 * (a - b) / b, 2, True), f'{num(ma, 0)}/{num(mb, 0)}',
                                  num(100 * (ma - mb) / mb, 2, True) if finite(ma, mb) else '--'])
    write_table(cost_rows, ['Setting', 'Shared rule', 'ms A3/shared', 'Latency delta (%)', 'MB A3/shared', 'Memory delta (%)'],
                'Resource overhead of candidate-specific gating relative to each shared rule', out, 'tab_shared_variants_costs')


def table_directional(results, out):
    rows = []
    for d, k in SETTINGS:
        cells = [pm(results.get('main', {}).get(key(m, d, k))).split(' ±')[0] for m in ('A3-query', 'A3-support', 'A3')]
        c = results.get('directional', {}).get(key(d, k), {})
        if any(x != '--' for x in cells):
            rows.append([*setting_label(d, k)] + cells + [ci(c.get('mutual_query')), ci(c.get('mutual_support'))])
    write_table(rows, ['Dataset', 'Setting', 'Query', 'Support', 'Mutual', 'Mutual-query [CI]', 'Mutual-support [CI]'],
                'Separately trained query-only, support-only and mutual configurations (joint F=12 intervals)', out, 'tab_directional_training')
    rows = []
    for k in (1, 5):
        q, m = resource(results, 'conv64', k, 'A3-query'), resource(results, 'conv64', k, 'A3')
        qm, mm = resource(results, 'conv64', k, 'A3-query', 'peak_mb'), resource(results, 'conv64', k, 'A3', 'peak_mb')
        if finite(q, m):
            rows.append([f'{k}-shot', 'All datasets', num(q), num(m), num(m - q, 1, True), num(qm, 0), num(mm, 0),
                         num(mm - qm, 0, True) if finite(qm, mm) else '--'])
    write_table(rows, ['Setting', 'Cost scope', 'Query ms', 'Mutual ms', 'Difference', 'Query MB', 'Mutual MB', 'Difference'],
                'Directional resource use (75-query episodes)', out, 'tab_directional_training_costs')


def table_strong(results, out):
    rows, ratios = [], []
    for d, k in SETTINGS:
        for ref in ('Bi-FRN', 'TDM', 'SFMR'):
            c = results.get('strong', {}).get(key(d, k), {}).get(ref)
            if not c:
                continue
            a, b = resource(results, 'conv64', k, 'A3'), resource(results, 'conv64', k, ref)
            ma, mb = resource(results, 'conv64', k, 'A3', 'peak_mb'), resource(results, 'conv64', k, ref, 'peak_mb')
            rows.append([*setting_label(d, k), ref, num(c['mean'], 1, True), interval(c), interval(c, True),
                         f'{num(a)}/{num(b)}', f'{num(ma, 0)}/{num(mb, 0)}'])
    write_table(rows, ['Dataset', 'Shot', 'Reference', 'Delta (pp)', 'Ordinary 95% CI', 'Joint CI', 'ms A3/ref', 'MB A3/ref'],
                'A3 versus Bi-FRN, TDM and SFMR (joint F=18 intervals)', out, 'tab_strong_baselines')
    for ref in ('Bi-FRN', 'TDM', 'SFMR'):
        for k in (1, 5):
            a, b = resource(results, 'conv64', k, 'A3'), resource(results, 'conv64', k, ref)
            ma, mb = resource(results, 'conv64', k, 'A3', 'peak_mb'), resource(results, 'conv64', k, ref, 'peak_mb')
            if finite(a, b):
                ratios.append([ref, f'{k}-shot', 'All datasets', num(a / b, 3), num(ma / mb, 3) if finite(ma, mb) else '--'])
    write_table(ratios, ['Baseline', 'Setting', 'Cost scope', 'A3/baseline latency', 'A3/baseline memory'],
                'Dimensionless A3/reference latency and memory ratios', out, 'tab_strong_baselines_ratios')


def table_backbone(results, out):
    b = results.get('backbone', {})
    rows = []
    for method in ('Bi-FRN', 'A2', 'C2-max', 'C2-mean', 'SFMR', 'A3-query', 'A3-support', 'A3'):
        cells = [pm(b.get('marginal', {}).get(key(method, k))).split(' ±')[0] for k in (1, 5)]
        if any(c != '--' for c in cells):
            ms = '/'.join(num(resource(results, 'resnet12', k, method)) for k in (1, 5))
            mb = '/'.join(num(resource(results, 'resnet12', k, method, 'peak_mb'), 0) for k in (1, 5))
            rows.append([method, 'ResNet-12'] + cells + [ms, mb])
    write_table(rows, ['Method', 'Backbone', '1-shot', '5-shot', 'Forward ms (1/5)', 'Peak MB (1/5)'], 'CUB-200-2011 ResNet-12 comparison', out,
                'tab_backbone_summary')
    rows = []
    for k in (1, 5):
        c = b.get('contrasts', {}).get(str(k), {})
        if any(c.values()):
            rows.append(['CUB-200-2011', f'{k}-shot', ci(c.get('a3_c2mean')), ci(c.get('a3_sfmr')), ci(c.get('mutual_query'))])
    write_table(rows, ['Dataset', 'Setting', 'A3-shared-mean [CI]', 'A3-SFMR [CI]', 'Mutual-query [CI]'],
                'ResNet-12 paired differences (pp), separate six-contrast family', out, 'tab_backbone_summary_contrasts')
    rows = []
    for d, k, backbone in [(d, k, 'conv64') for d, k in SETTINGS] + [('cub', 1, 'resnet12'), ('cub', 5, 'resnet12')]:
        src = results.get('main', {}) if backbone == 'conv64' else {key(m, 'cub', kk): v for kk in (1, 5) for m in ('Bi-FRN', 'A2', 'C2-max', 'A3')
                                                                     for v in [b.get('marginal', {}).get(key(m, kk))] if v}
        cells = [pm(src.get(key(m, d, k))).split(' ±')[0] for m in ('Bi-FRN', 'A2', 'C2-max', 'A3')]
        if any(c != '--' for c in cells):
            rows.append([*setting_label(d, k), 'Conv-64' if backbone == 'conv64' else 'ResNet-12'] + cells)
    write_table(rows, ['Dataset', 'Setting', 'Backbone', 'Bi-FRN', 'A2', 'C2', 'A3'], 'Reconstruction-baseline and backbone analysis', out, 'tab_backbone')
    rows = [['CUB-200-2011', f'{k}-shot', 'ResNet-12', num(c['mean'], 1, True), interval(c), interval(c, True)]
            for k in (1, 5) for c in [b.get('bifrn', {}).get(str(k))] if c]
    write_table(rows, ['Dataset', 'Setting', 'Backbone', 'Delta (pp)', '95% CI', 'Simultaneous CI'], 'ResNet-12 A3-Bi-FRN differences (F=2)', out,
                'tab_backbone_bifrn')
    rows = [['CUB-200-2011', f'{k}-shot', num(i['conv64_gain'], 1, True), num(i['resnet12_gain'], 1, True),
             num(i['resnet12_gain'] - i['conv64_gain'], 1, True), num(i['I']['mean'], 2, True), interval(i['I']), interval(i['I'], True)]
            for k in (1, 5) for i in [b.get('interaction', {}).get(str(k))] if i]
    write_table(rows, ['Dataset', 'Setting', 'Conv-64 gain (pp)', 'ResNet-12 gain (pp)', 'Difference (pp)', 'I (pp)', 'Ordinary 95% interval',
                       'Two-shot simultaneous interval'], 'Cross-backbone interaction of the A3-shared-mean increment', out, 'tab_followup_backbone')


def table_confirm(results, out):
    rows = []
    for method, name in CONFIRM_NAMES.items():
        c = results.get('confirm', {}).get(method)
        if not c:
            continue
        v = c.get('versus')
        rows.append([method, name, pm(c['marginal']), f'{v["mean"]:+.1f} {interval(v, digits=1)}' if v and v['n'] > 1 else '--',
                     f'{v["positive"]}/{v["n"]}' if v else '--'])
    write_table(rows, ['Code', 'Variant', 'Acc. (%)', 'Delta A3-X (pp)', 'Seeds with Delta>0'],
                'Confirmation-pool results on V_conf (CUB-200-2011, 5-way 1-shot, Conv-64)', out, 'tab_confirm',
                note='Ordinary paired 95% intervals, not family-wise adjusted.')


def table_practical(results, out):
    rows = [[num(r['margin'], 1), f'{r["mean_at_least"]}/6', f'{r["lower_above"]}/6', f'{r["within_budget"]}/6'] for r in results.get('practical', [])]
    if any(c.get('a3_c2') for c in results.get('c2_test', {}).values()):
        write_table(rows, ['Margin delta (pp)', 'Mean >= delta', 'Lower CI > delta', 'Also within budget'],
                    'Post hoc practical-margin sensitivity of A3 versus C2-max', out, 'tab_practical_effect')


def table_c2_test(results, out):
    rows, seeds = [], []
    for d, k in SETTINGS:
        c = results.get('c2_test', {}).get(key(d, k), {})
        a, b = c.get('c2_a2'), c.get('a3_c2')
        if a or b:
            rows.append([*setting_label(d, k), num(a['mean'], 1, True) if a else '--', interval(a), interval(a, True),
                         num(b['mean'], 1, True) if b else '--', interval(b)])
        if b:
            seeds.append([*setting_label(d, k), ', '.join(f'{x:+.2f}' for x in b['deltas']), f'{b["positive"]}/{b["n"]}', interval(b, True)])
    write_table(rows, ['Dataset', 'Setting', 'C2-A2', '95% CI', 'Simultaneous CI', 'A3-C2', '95% CI'], 'Paired increments (pp) across training seeds',
                out, 'tab_c2_test')
    write_table(seeds, ['Dataset', 'Setting', 'Delta_s (pp)', 'Seeds > 0', 'Simultaneous CI'], 'A3-C2-max differences in seed order', out,
                'tab_c2_test_seeds')


def table_transitions_trained(results, out):
    rows = []
    for d, k in SETTINGS:
        a, b = results.get('transitions', {}).get(key(d, k, 'A2')), results.get('transitions', {}).get(key(d, k, 'C2-mean'))
        if a or b:
            rows.append([*setting_label(d, k)] + [num(x[f], 2) if x else '--' for x in (a,) for f in ('corrected', 'new_errors')] +
                        [num(a['net'], 1, True) if a else '--'] + [num(x[f], 2) if x else '--' for x in (b,) for f in ('corrected', 'new_errors')] +
                        [num(b['net'], 1, True) if b else '--'])
    write_table(rows, ['Dataset', 'Setting', 'A3 vs A2 corrected', 'A3 vs A2 new errors', 'A3 vs A2 net', 'A3 vs C2-mean corrected',
                       'A3 vs C2-mean new errors', 'A3 vs C2-mean net'], 'Benchmark decision contingencies (% of all queries)', out, 'tab_trained_transitions')


def table_robust(results, out):
    r = results.get('robust', {})
    if not r:
        return
    methods = [m for m in ('FRN', 'Bi-FRN', 'A2', 'C2-max', 'A3') if m in r.get('drops', {})]
    rows = [['Clean accuracy'] + [num(r['clean'].get(m, {}).get('mean')) for m in methods] + ['--', '--']]
    for family in list(FAMILIES) + ['mean', 'mean_no_background']:
        values = {m: r['drops'][m].get(family, {}).get('mean') for m in methods}
        label = FAMILY_NAMES.get(family, {'mean': 'Mean drop', 'mean_no_background': 'Mean, no background'}.get(family))
        rows.append([label] + [num(values[m], 2 if family.startswith('mean') else 1) for m in methods] +
                    [num(values.get('A3') - values.get('A2'), 1, True) if finite(values.get('A3'), values.get('A2')) else '--',
                     num(values.get('A3') - values.get('C2-max'), 1, True) if finite(values.get('A3'), values.get('C2-max')) else '--'])
    write_table(rows, ['Condition'] + methods + ['Drop A3-A2', 'Drop A3-C2'], 'Query-only interventions (CUB, 5-way 1-shot): clean accuracy and drops (pp)',
                out, 'tab_robust')
    rows = [[name.replace('C2-max', 'C2') + ' drop', ', '.join(f'{x:+.2f}' for x in c['deltas']), num(c['mean'], 2, True), interval(c), interval(c, True)]
            for name, c in r.get('contrasts', {}).items()]
    write_table(rows, ['Background contrast', 'Delta_s (pp)', 'Mean', '95% CI', 'Simultaneous CI'], 'Background-drop contrasts (F=3)', out,
                'tab_robust_background')


def table_diag(results, out):
    rows = []
    names = {'A1': 'A1 (balanced OT)', 'A2': 'A2 (partial OT)', 'C2-max': 'C2 (shared rival gate)', 'A3': 'CaPT-MR (A3)'}
    for method, name in names.items():
        d = results.get('diagnostics', {}).get(method)
        if d:
            rows.append([name, num(d['box_mass'], 2), num(d['effective_rows'], 1), num(d['rerouting'], 2), num(d['margin_x100'], 1)])
    write_table(rows, ['Model', 'Box mass', 'Effective rows', 'Rerouting', 'Margin (x1e-2)'], 'True-class transport diagnostics (CUB test, 5-way 1-shot)',
                out, 'tab_diag')
    rows = []
    res = results.get('resources', {})
    for method in ('ProtoNet', 'FRN', 'Bi-FRN', 'DeepEMD', 'A1', 'A2', 'C2-max', 'A3'):
        lat = res.get('latency', {}).get(key('conv64', 1, method))
        model = res.get('models', {}).get(key('conv64', 1, method))
        if not lat:
            continue
        sweeps = '--' if not finite(lat['sweeps_median']) else f'{lat["sweeps_median"]:.0f}/{lat["sweeps_p95"]:.0f}/{lat["sweeps_max"]:.0f}'
        rows.append([method, num(model['params'] / 1000.0, 1) if model else '--', sweeps, num(lat['cap_percent'], 1) if finite(lat['sweeps_median']) else '--',
                     num(lat['median']), num(model['train_ms']) if model else '--', num(lat['peak_mb'], 0)])
    write_table(rows, ['Method', 'Params (K)', 'Sweeps', 'Cap (%)', 'Fwd (ms)', 'Train (ms)', 'Mem. (MB)'], 'Transport cost (5-way 1-shot)', out,
                'tab_diag_cost')
    rows = []
    for method in ('A2', 'C2-max', 'A3'):
        s = res.get('stages', {}).get(key('conv64', 1, method))
        if s:
            rows.append([method, num(s['encoder']), num(s['costs_gate']), num(s['solver']), num(s['recon']), num(s['head']),
                         num(resource(results, 'conv64', 1, method))])
    write_table(rows, ['Method', 'Encoder', 'Costs+gate', 'OT solver', 'Recon.', 'Head', 'Total'], 'Device-resident stage times (ms)', out,
                'tab_diag_stages')


def table_efficiency(results, out):
    rows = []
    for d, k in SETTINGS:
        c = results.get('c2_test', {}).get(key(d, k), {}).get('a3_c2')
        a, b = resource(results, 'conv64', k, 'A3'), resource(results, 'conv64', k, 'C2-max')
        if c and finite(a, b):
            rows.append([*setting_label(d, k), num(c['mean'], 1, True), num(b), num(a), num(a / b, 3),
                         num(resource(results, 'conv64', k, 'C2-max', 'peak_mb'), 0), num(resource(results, 'conv64', k, 'A3', 'peak_mb'), 0)])
    write_table(rows, ['Dataset', 'Setting', 'Delta (pp)', 'C2 (ms)', 'A3 (ms)', 'A3/C2', 'C2 (MB)', 'A3 (MB)'], 'Accuracy-efficiency comparisons', out,
                'tab_efficiency')


def table_followup_cost(results, out):
    res = results.get('resources', {})
    rows = []
    for method in ('Bi-FRN', 'TDM', 'SFMR', 'C2-max', 'C2-mean', 'C2-weighted', 'A3-query', 'A3-support', 'A3'):
        one, five = res.get('latency', {}).get(key('conv64', 1, method)), res.get('latency', {}).get(key('conv64', 5, method))
        if one or five:
            sweeps = f'{one["sweeps_median"]:.0f}/{one["sweeps_p95"]:.0f}' if one and finite(one['sweeps_median']) else '--'
            rows.append([method, num(one['median']) if one else '--', num(five['median']) if five else '--', num(one['peak_mb'], 0) if one else '--',
                         num(five['peak_mb'], 0) if five else '--', sweeps])
    write_table(rows, ['Method', '1-shot (ms)', '5-shot (ms)', '1-shot (MB)', '5-shot (MB)', 'OT sweeps'], 'Conv-64 resource use (5-way, 75-query episodes)',
                out, 'tab_followup_cost')
    rows = []
    for method in ('Bi-FRN', 'SFMR', 'A2', 'C2-mean', 'A3'):
        for k in (1, 5):
            lat = res.get('latency', {}).get(key('conv64', k, method))
            if lat:
                rows.append([f'{method}, {k}-shot', num(lat['median'], 2), num(lat['q1'], 2), num(lat['q3'], 2), num(lat['p95'], 2)])
    write_table(rows, ['Method, setting', 'Median ms', 'Q1 ms', 'Q3 ms', 'p95 ms'], 'Latency quantiles', out, 'tab_followup_cost_quantiles')
    rows = []
    for k in (1, 5):
        for ref in ('C2-mean', 'Bi-FRN', 'SFMR'):
            d = res.get('differences', {}).get(key('conv64', k, 'A3', ref))
            if d:
                rows.append([f'A3-{ref}', f'{k}-shot', num(d['median'], 2, True), num(d['q1'], 2, True), num(d['q3'], 2, True), num(d['p95'], 2, True)])
    write_table(rows, ['Comparison', 'Setting', 'Median ms', 'Q1 ms', 'Q3 ms', 'p95 ms'], 'Quantiles of matched timing differences', out,
                'tab_followup_cost_differences')


def table_operating_region(results, out):
    rows, dom = [], []
    for d, k in SETTINGS:
        r = results.get('operating_region', {}).get(key(d, k))
        if not r or not r.get('best_shared'):
            continue
        rows.append([*setting_label(d, k), ci(r['best_shared']), 'Yes' if r['useful'] else 'No', ci(r.get('mutual_query'))])
        dom.append([*setting_label(d, k), r['dominance'].get('Bi-FRN') or '--', r['dominance'].get('SFMR') or '--'])
    write_table(rows, ['Dataset', 'Setting', 'A3-best shared [CI]', 'Useful gate?', 'Mutual-query [CI]'], 'Conditional utility analysis', out,
                'tab_operating_region')
    write_table(dom, ['Dataset', 'Setting', 'A3 versus Bi-FRN', 'A3 versus SFMR'], 'Point dominance in accuracy, latency and memory', out,
                'tab_operating_region_dominance')


def computed_tables(ws, out):
    probe = ws.probe
    if (probe / 'harder_summary.csv').exists():
        summary = pd.read_csv(probe / 'harder_summary.csv')
        rows = [[r.rule, num(r.accuracy), str(r.errors), num(r.median_margin, 4, True), f'[{r.margin_q05:+.4f}, {r.margin_q95:+.4f}]']
                for r in summary.itertuples()]
        write_table(rows, ['Rule', 'Accuracy (%)', 'Errors / 100', 'Median margin', '[5%,95%] margin'],
                    'Computed harder fixed-feature probe (allocation controls, mutual scoring)', out, 'tab_computed_mechanisms')
    if (probe / 'directions.csv').exists():
        names = {'query': 'Query-only R_q', 'support': 'Support-only R_s', 'mutual': 'Mutual (R_q+R_s)/2'}
        rows = [[names[r.score], num(r.accuracy), str(r.errors), num(r.median_margin, 4, True), f'[{r.margin_q05:+.4f}, {r.margin_q95:+.4f}]']
                for r in pd.read_csv(probe / 'directions.csv').itertuples()]
        write_table(rows, ['Score', 'Accuracy (%)', 'Errors / 100', 'Median margin', '[5%,95%] margin'],
                    'Computed one-coupling scoring diagnostic (same A3 couplings rescored by direction)', out, 'tab_directions')
    if (probe / 'transitions.csv').exists():
        rows = [[f'{r.rule} / {r.reference}', f'{r.corrected}/{r.reference_errors}', f'{r.new_errors}/{r.reference_correct}',
                 f'{r.margin_rises}/{r.queries}', num(r.median_margin_change, 5, True)] for r in pd.read_csv(probe / 'transitions.csv').itertuples()]
        write_table(rows, ['Rule / reference', 'Corrected', 'New errors', 'Margin rises', 'Median delta margin'], 'Computed prediction transitions', out,
                    'tab_transitions')
    if (probe / 'ambiguity_accuracy.csv').exists():
        acc = pd.read_csv(probe / 'ambiguity_accuracy.csv')
        rows = []
        for (bank, band), part in acc.groupby(['bank', 'band'], sort=False):
            values = dict(zip(part['method'], part['accuracy']))
            rows.append([bank, band, str(int(part['queries'].iloc[0]))] + [num(values.get(m)) for m in ('A2', 'C2-max', 'C2-mean', 'C2-weighted', 'A3')])
        write_table(rows, ['Bank', 'Band', 'Queries', 'A2', 'C2-max', 'C2-mean', 'C2-weighted', 'A3'],
                    'Computed accuracies (%) in calibration-defined bands', out, 'tab_ambiguity_accuracy')
        eff = pd.read_csv(probe / 'ambiguity_effects.csv')
        rows = []
        for (contrast, rule), part in eff.groupby(['contrast', 'rule'], sort=False):
            cells = []
            for bank in ('Existing', 'New'):
                r = part[part['bank'] == bank]
                cells.append(f'{r["difference"].iloc[0]:+.1f} [{r["lo"].iloc[0]:+.1f}, {r["hi"].iloc[0]:+.1f}]' if len(r) else '--')
            rows.append([contrast, rule] + cells)
        write_table(rows, ['Population/contrast', 'Shared rule', 'Existing: difference [interval]', 'New: difference [interval]'],
                    'Computed A3 minus shared-rule differences with Bonferroni-adjusted episode-cluster bootstrap intervals (30 contrasts)', out,
                    'tab_ambiguity_effects')
    if (probe / 'convergence.csv').exists():
        rows = [[r.method, r.setting, str(r.solves), f'{r.sweeps_median:.0f}/{r.sweeps_p95:.1f}/{r.sweeps_max}', str(r.cap_hits),
                 f'{r.max_feasibility:.2e}', f'{r.max_reference_l1:.2e}'] for r in pd.read_csv(probe / 'convergence.csv').itertuples()]
        write_table(rows, ['Method', 'Setting', 'Solves', 'Sweeps med/p95/max', 'Cap hits', 'Max feasibility', 'Max reference L1'],
                    'Computed convergence on a small matched procedural-feature sample', out, 'tab_computed_cost_convergence')
    if (probe / 'directional_kernel.csv').exists():
        rows = [[r.setting, r.scorer.capitalize(), str(r.couplings), num(r.median_ms, 3), f'[{r.q1_ms:.3f}, {r.q3_ms:.3f}]', num(r.p95_ms, 3),
                 num(r.traced_kib, 1)] for r in pd.read_csv(probe / 'directional_kernel.csv').itertuples()]
        write_table(rows, ['Setting', 'Scorer', 'Couplings', 'Median ms', '[Q1,Q3] ms', 'p95 ms', 'Traced KiB'],
                    'Measured local CPU residual-kernel costs on cached procedural A3 couplings', out, 'tab_directional_kernel')
    v = load_json_safe(ws.verify / 'verify_summary.json')
    p = load_json_safe(probe / 'probe_summary.json')
    rows = []
    if v:
        rows += [
            ['Residual-dispersion identity P=R+V (max error)', f'{v["identity"]["instances"]} instances', f'{v["identity"]["max_identity_error"]:.1e}'],
            ['P = 2<T,C>/rho (max error)', f'{v["identity"]["instances"]} instances', f'{v["identity"]["max_cost_error"]:.1e}'],
            ['Dispersion at exact LP vertices (max)', 'simplex solutions', f'{v["identity"]["max_vertex_dispersion"]:.1e}'],
            ['Full-mass gate neutrality (max entry change)', f'{v["full_mass"]["problems"]} problems', f'{v["full_mass"]["max_abs_diff"]:.1e}'],
            ['Sensitivity chain (largest positive excess)', f'{v["sensitivity"]["pairs"]} pairs', f'{v["sensitivity"]["max_positive_excess"]:.2e}'],
            ['Constant row bias (max entry change)', f'{v["sensitivity"]["pairs"]} pairs', f'{v["sensitivity"]["max_constant_shift_diff"]:.2e}'],
            ['Full-mass sensitivity pairs (max L1)', f'{v["sensitivity"]["full_mass_pairs"]} pairs', f'{v["sensitivity"]["full_mass_max_l1"]:.2e}'],
            ['Tied-cost plan T=rho/4 (max entry error)', '6 settings', f'{v["tied_cost"]["max_entry_error"]:.1e}'],
            ['Median normalized row-marginal change', f'{v["allocation"]["problems"]} problems', num(100 * v['allocation']['median_row_change'], 1) + '%'],
            ['Median reallocated transported mass', f'{v["allocation"]["problems"]} problems', num(100 * v['allocation']['median_reallocated'], 1) + '%'],
            ['Median Spearman(gate, row-mass change)', f'{v["allocation"]["problems"]} problems', num(v['allocation']['median_spearman'], 2)],
            ['Participating rows changing destination', 'pooled rows', num(100 * v['allocation']['rerouted_fraction'], 1) + '%'],
            ['Object-row share, true pairs (no gate -> gate)', 'true-class pairs',
             f'{v["allocation"]["object_share_plain"]:.2f} -> {v["allocation"]["object_share_gated"]:.2f}'],
            ['Plan-only stopping: problems above 1e-5 violation', 'gated problems', num(100 * v['stopping']['plan_only_violation_fraction'], 1) + '%'],
            ['Combined stopping: median sweeps (no gate / gate)', 'all problems',
             f'{v["stopping"]["plain_sweeps_median"]:.0f} / {v["stopping"]["gated_sweeps_median"]:.0f}'],
        ]
    if p:
        e = p['easy']
        rows += [
            ['Easy probe: wrong candidates with lower residual', f'{e["candidate_problems"] - e["queries"]} problems', num(100 * e['wrong_lowered_fraction'], 1) + '%'],
            ['Easy probe: true candidates with lower residual', f'{e["queries"]} problems', num(100 * e['true_lowered_fraction'], 1) + '%'],
            ['Easy probe: accuracy A2 / A3', f'{e["queries"]} queries', f'{e["accuracy_A2"]:.1f} / {e["accuracy_A3"]:.1f}'],
            ['Harder probe: anchored bound (max excess)', f'{p["harder"]["anchored_pairs"]} pairs', f'{p["harder"]["anchored_bound_max_excess"]:.2e}'],
        ]
    write_table(rows, ['Check', 'Scope', 'Value'], 'Computed transport verification on procedural descriptors', out, 'tab_verification',
                note='Finite-tolerance checks of the stated transport properties; they do not provide classification guarantees.')


def table_setup(results, out):
    cfg = results.get('config', {})
    keys = ['datasets', 'shots', 'seeds', 'image_size', 'n_way', 'train_episodes', 'train_queries', 'test_episodes', 'test_queries', 'val_every',
            'lr', 'momentum', 'weight_decay', 'lr_milestones', 'jitter', 'tau_init', 'rho', 'entropy', 'lambda_g', 'eta', 't_m', 'kappa',
            'ot_tol', 'ot_feas_tol', 'ot_max_iter', 'use_tuned', 'seed']
    write_table([[k, str(cfg.get(k))] for k in keys if k in cfg], ['Setting', 'Value'], 'Configuration of this run', out, 'tab_setup')


def figure_toy(ws, out):
    path = ws.verify / 'toy_pair.npz'
    if not path.exists():
        return False
    data = np.load(path)
    gate, obj = data['gate'], data['object'].astype(bool)
    order = np.argsort(-gate, kind='stable')
    length = len(gate)
    fig = plt.figure(figsize=(11.5, 2.6))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.7, 1, 1, 1.6], wspace=0.35)
    ax = fig.add_subplot(grid[0])
    ax.barh(np.arange(length), gate[order], color=[ORANGE if obj[i] else LIGHT for i in order], height=0.8)
    ax.set_ylim(length - 0.5, -0.5)
    ax.set_xlabel(r'gate $g(i,c)$')
    ax.set_ylabel(r'query patch $i$ (sorted by $g$)')
    ax.set_title('(a)', loc='left', fontsize=11)
    vmax = max(data['plan_plain'].max(), data['plan_gated'].max())
    for j, (name, title) in enumerate((('plan_plain', rf'(b) $T$, no gate, $\rho={float(data["rho"]):g}$'),
                                       ('plan_gated', rf'(c) $T$, gated, $\rho={float(data["rho"]):g}$'))):
        ax = fig.add_subplot(grid[j + 1])
        ax.imshow(data[name][order], cmap='Greens', vmin=0, vmax=vmax, aspect='auto', interpolation='nearest')
        ax.set_xlabel(r'support patch $j$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc='left', fontsize=11)
    ax = fig.add_subplot(grid[3])
    x = np.arange(length)
    rho = float(data['rho'])
    ax.plot(x, data['plan_plain'].sum(1)[order] * length, color=GREY, marker='s', ms=4, lw=1.4, label=rf'no gate, $\rho={rho:g}$')
    ax.plot(x, data['plan_gated'].sum(1)[order] * length, color=ORANGE, marker='o', ms=4, lw=1.4, label=rf'gated, $\rho={rho:g}$')
    ax.plot(x, data['full_plain'].sum(1)[order] * length, color=BLUE, lw=2.2, label=r'no gate, $\rho=1$')
    ax.plot(x, data['full_gated'].sum(1)[order] * length, color='black', lw=1.2, ls='--', label=r'gated, $\rho=1$')
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r'query patch $i$ (sorted by $g$)')
    ax.set_ylabel(r'row mass $r_i/(1/L)$')
    ax.set_title('(d) row participation', loc='left', fontsize=11)
    ax.grid(True)
    ax.legend(loc='lower left', fontsize=9, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
    save(fig, out, 'fig_toy_transport')
    return True


def figure_harder_probe(ws, out):
    path = ws.probe / 'harder_queries.csv'
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    methods = [m for m in PROBE_STYLES if m in set(frame['method'])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.0), gridspec_kw={'wspace': 0.3})
    for method in methods:
        margin = np.sort(frame.loc[frame['method'] == method, 'margin'].to_numpy())
        ax1.step(margin, np.arange(1, len(margin) + 1) / len(margin), where='post', color=PROBE_STYLES[method], lw=1.6, label=method)
    ax1.axvline(0, color='black', ls=':', lw=1)
    ax1.set_xlabel('best-wrong score minus true score')
    ax1.set_ylabel('empirical cumulative fraction')
    ax1.legend(ncol=2, fontsize=8.5, loc='upper left')
    ax1.grid(True)
    positions, data, colours = [], [], []
    for i, method in enumerate(methods):
        part = frame[frame['method'] == method]
        for offset, column, colour in ((-0.18, 'true_residual', BLUE), (0.18, 'best_wrong_residual', ORANGE)):
            positions.append(i + offset)
            data.append(part[column].to_numpy())
            colours.append(colour)
    boxes = ax2.boxplot(data, positions=positions, widths=0.3, showfliers=False, whis=1.5, patch_artist=True, medianprops={'color': 'black'})
    for patch, colour in zip(boxes['boxes'], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.85)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=35, ha='right')
    ax2.set_ylabel('reconstruction residual')
    ax2.legend(handles=[Line2D([], [], color=BLUE, lw=6, label='True'), Line2D([], [], color=ORANGE, lw=6, label='Best wrong')], loc='lower left',
               fontsize=9, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
    ax2.grid(True, axis='y')
    save(fig, out, 'fig_harder_gate_probe')
    return True


def figure_coverage(results, args, out):
    cov, frozen, retrained = results.get('coverage', {}), results.get('kappa_frozen', {}), results.get('kappa_retrained', {})
    if not cov and not frozen:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.0), gridspec_kw={'wspace': 0.35})
    for method, label in (('A2', 'A2 (no gate)'), ('A3', 'A3 (rival gate)')):
        points = sorted((float(k.split('|')[1]), v) for k, v in cov.items() if k.split('|')[0] == method)
        if points:
            colour, marker, _ = METHOD_STYLES[method]
            ax1.errorbar([p[0] for p in points], [p[1]['mean'] for p in points], yerr=[np.nan_to_num(p[1]['sd']) for p in points], color=colour,
                         marker=marker, ms=6, lw=1.8, capsize=4, label=label)
    ax1.axvline(args.rho, color='black', ls='--', lw=1)
    ax1.text(args.rho + 0.01, 0.05, r'primary $\rho$', transform=ax1.get_xaxis_transform(), fontsize=9)
    ax1.set_xlabel(r'coverage $\rho$')
    ax1.set_ylabel('accuracy (%)')
    ax1.set_title('(a) gate effect vs. coverage', loc='left', fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True)
    if frozen:
        items = sorted((float(k), v) for k, v in frozen.items())
        ax2.plot([k for k, _ in items], [v['mean'] for _, v in items], color=ORANGE, marker='o', ms=5, lw=1.8, label=r'A3, frozen plans, sweep $\kappa$')
    for name, colour, marker, label, x in (('A3_kappa', ORANGE, 'D', rf'A3 retrained, $\kappa={args.kappa_retrain:g}$', args.kappa_retrain),
                                           ('A3_small_entropy', PURPLE, '^', rf'A3, $\kappa=0$, $\varepsilon={args.small_entropy:g}$', 0.0),
                                           ('A2_kappa', GREY, 's', rf'A2 retrained, $\kappa={args.kappa_retrain:g}$', args.kappa_retrain)):
        if name in retrained:
            face = 'white' if name == 'A3_kappa' else colour
            ax2.plot([x], [retrained[name]['mean']], marker=marker, ms=7, ls='none', color=colour, markerfacecolor=face, label=label)
    ax2.set_xlabel(r'dispersion weight $\kappa$')
    ax2.set_ylabel('accuracy (%)')
    ax2.set_title('(b) residual-score diagnostic', loc='left', fontsize=11)
    ax2.legend(loc='lower left', fontsize=8.5)
    ax2.grid(True)
    save(fig, out, 'fig_coverage_dispersion')
    return True


def figure_manyway(results, args, out):
    many = results.get('manyway', {})
    if not many:
        return False
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.0), gridspec_kw={'wspace': 0.3})
    for method in ('FRN', 'Bi-FRN', 'A2', 'C2-max', 'A3'):
        points = [(w, many[key(method, w)]) for w in args.ways if key(method, w) in many]
        if not points:
            continue
        colour, marker, ls = METHOD_STYLES[method]
        ax1.plot([p[0] for p in points], [p[1]['mean'] for p in points], color=colour, marker=marker, ls=ls, lw=1.8, ms=5, label=FIG_NAMES[method])
        ax2.plot([p[0] for p in points], [p[1]['ms_per_query'] for p in points], color=colour, marker=marker, ls=ls, lw=1.8, ms=5,
                 label=FIG_NAMES[method])
    for ax in (ax1, ax2):
        ax.set_xscale('log')
        ax.set_xticks(args.ways)
        ax.set_xticklabels([str(w) for w in args.ways])
        ax.minorticks_off()
        ax.set_xlabel(r'number of ways $N$')
        ax.grid(True)
    ax1.set_ylabel('accuracy (%)')
    ax1.set_title('(a) accuracy', loc='left', fontsize=11)
    ax1.legend(loc='lower left', fontsize=9)
    diffs = [(w, many[key('A3-C2', w)]['mean']) for w in args.ways if key('A3-C2', w) in many]
    if diffs:
        inset = ax1.inset_axes([0.58, 0.55, 0.38, 0.38])
        inset.bar(range(len(diffs)), [d[1] for d in diffs], color=ORANGE)
        inset.set_xticks(range(len(diffs)))
        inset.set_xticklabels([str(d[0]) for d in diffs], fontsize=7)
        inset.tick_params(labelsize=7)
        inset.set_title('A3-C2 (pp)', fontsize=8)
    ax2.set_yscale('log')
    ax2.set_ylabel('ms per query')
    ax2.set_title('(b) inference cost', loc='left', fontsize=11)
    ax2.legend(loc='upper left', fontsize=9)
    save(fig, out, 'fig_many_way')
    return True


def figure_perturbations(ws, args, out):
    path = ws.split_path('cub')
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    images, boxes = np.load(ws.cache_paths('cub', args.image_size)[0], mmap_mode='r'), np.load(ws.cache_paths('cub', args.image_size)[1])
    test = np.flatnonzero(frame['pool'].to_numpy() == 'test')
    idx = int(test[0])
    donor = int(np.flatnonzero(frame['class_id'].to_numpy() != frame['class_id'].iloc[idx])[0])
    rng = np.random.default_rng(args.seed)
    titles = {'blur': 'Gaussian blur', 'jpeg': 'JPEG', 'brightness': 'brightness', 'contrast': 'contrast', 'noise': 'Gaussian noise',
              'background': 'background repl.'}
    panels = [('clean', np.asarray(images[idx]))]
    for family in FAMILIES:
        panels.append((titles[family],
                       apply_condition(np.asarray(images[idx]), f'{family}-2', args, rng=rng, box=boxes[idx], donor=np.asarray(images[donor]),
                                       donor_box=boxes[donor])))
    fig, axes = plt.subplots(1, len(panels), figsize=(1.45 * len(panels), 1.7))
    for ax, (title, image) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    save(fig, out, 'fig_perturbations')
    return True


def load_display(path, size):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB').resize((size, size), Image.BILINEAR))


def figure_qualitative(ws, out):
    path = ws.qualitative / 'selection.json'
    if not path.exists():
        return False
    selection = load_json(path)
    picks = selection['picks']
    if not picks:
        return False
    arrays = np.load(ws.qualitative / 'participation.npz')
    size = selection['size']
    fig, axes = plt.subplots(len(picks), 4, figsize=(7.6, 2.0 * len(picks)), squeeze=False)
    frames = {}
    for r, pick in enumerate(picks):
        dataset = pick['dataset']
        if dataset not in frames:
            frames[dataset] = pd.read_csv(ws.split_path(dataset))
        frame = frames[dataset]
        classes = [int(c) for c in str(pick['classes']).split(',')]
        names = dict(zip(frame['class_id'], frame['class_name']))
        true_name = names[classes[int(pick['label'])]]
        query = load_display(frame['path'].iloc[int(pick['query_index'])], size)
        support = load_display(frame['path'].iloc[int(pick['support_index'])], size)
        panels = [(query, None, f'true: {true_name}'), (support, None, f'support: {true_name}')]
        for method in ('A2', 'A3'):
            pred = int(pick[f'pred_{method}'])
            verdict = 'correct' if pred == int(pick['label']) else 'wrong'
            panels.append((query, arrays[f'{pick["key"]}_{method}'], f'pred: {names[classes[pred]]}\n({verdict})'))
        for c, (image, participation, caption) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(image, extent=(0, size, size, 0))
            if participation is not None:
                grid = int(round(np.sqrt(len(participation))))
                mappable = ax.imshow(participation.reshape(grid, grid), cmap='magma', vmin=0, vmax=1, alpha=0.55, extent=(0, size, size, 0),
                                     interpolation='nearest')
                for t in np.linspace(0, size, grid + 1):
                    ax.axhline(t, color='white', lw=0.4, alpha=0.6)
                    ax.axvline(t, color='white', lw=0.4, alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(caption, fontsize=7.5)
            if r == 0:
                ax.set_title(('Query', 'Support of true class', 'A2: class-agnostic partial OT', 'A3: CaPT-MR')[c], fontsize=8.5)
        label = DATASET_NAMES[dataset] if pick['row'] == 'correct' else f'Failure ({DATASET_NAMES[dataset].split("-")[0]})'
        axes[r, 0].set_ylabel(label, fontsize=9, fontweight='bold')
    fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02, label='patch participation (same scale for A2 and A3)')
    save(fig, out, 'fig_qualitative_transport')
    return True


def figure_datasets(ws, out):
    available = [d for d in DATASET_NAMES if ws.split_path(d).exists()]
    if not available:
        return False
    fig, axes = plt.subplots(len(available), 3, figsize=(6.6, 1.9 * len(available)), squeeze=False)
    for r, dataset in enumerate(available):
        frame = pd.read_csv(ws.split_path(dataset))
        for c, class_id in enumerate(sorted(frame['class_id'].unique())[:3]):
            row = frame[frame['class_id'] == class_id].iloc[0]
            with Image.open(row['path']) as image:
                image = image.convert('RGB')
                w, h = image.size
                target = min(w, int(h * 4 / 3))
                left = (w - target) // 2
                crop_h = int(target * 3 / 4)
                top = (h - crop_h) // 2
                axes[r, c].imshow(np.asarray(image.crop((left, top, left + target, top + crop_h))))
            axes[r, c].set_title(row['class_name'], fontsize=8)
            axes[r, c].axis('off')
        axes[r, 0].text(-0.08, 0.5, DATASET_NAMES[dataset], transform=axes[r, 0].transAxes, rotation=90, ha='right', va='center', fontsize=9)
    save(fig, out, 'fig_dataset_examples')
    return True


def stage_report(args, ws):
    results = load_json_safe(ws.metrics / 'results.json')
    out = ws.report
    with plt.rc_context(STYLE):
        made = {'toy': figure_toy(ws, out), 'harder probe': figure_harder_probe(ws, out), 'coverage': figure_coverage(results, args, out),
                'many-way': figure_manyway(results, args, out), 'perturbations': figure_perturbations(ws, args, out),
                'qualitative': figure_qualitative(ws, out), 'datasets': figure_datasets(ws, out)}
    computed_tables(ws, out)
    if results:
        table_main(results, out)
        table_components(results, out)
        table_shared(results, out)
        table_directional(results, out)
        table_strong(results, out)
        table_backbone(results, out)
        table_confirm(results, out)
        table_practical(results, out)
        table_c2_test(results, out)
        table_transitions_trained(results, out)
        table_robust(results, out)
        table_diag(results, out)
        table_efficiency(results, out)
        table_followup_cost(results, out)
        table_operating_region(results, out)
        table_setup(results, out)
    skipped = [k for k, v in made.items() if not v]
    log(f'report: figures and tables written to {out}' + (f'; figures without inputs on this run: {", ".join(skipped)}' if skipped else ''))
