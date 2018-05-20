from pathlib import Path

import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import widgetbox, column
from bokeh.models import ColumnDataSource, CDSView, CustomJS, GroupFilter, HoverTool
from bokeh.models.widgets import RadioButtonGroup, Select
from bokeh.palettes import Category10
from bokeh.plotting import figure
from tqdm import tqdm
import numpy as np

COLUMN_MAPPING = dict(
    online='online_mean@10',
    online_std='online_std@10',
    offline='offline_mean@10',
    offline_std='offline_std@10'
)

BENCHMARK_DATA = dict(
    per=dict(        
        HP2003=dict(online=764.4, offline=0.782),
        NP2003=dict(online=699.5, offline=0.719),
        TD2003=dict(online=312.2, offline=0.327),
        HP2004=dict(online=723.3, offline=0.751),
        NP2004=dict(online=719.9, offline=0.719),
        TD2004=dict(online=298.9, offline=0.333),
        MQ2007=dict(online=412.5, offline=0.406),
        MQ2008=dict(online=523.2, offline=0.493),
        OHSUMED=dict(online=494.8, offline=0.456)
    ),
    nav=dict(        
        HP2003=dict(online=701.2, offline=0.764),
        NP2003=dict(online=637.6, offline=0.711),
        TD2003=dict(online=272.5, offline=0.315),
        HP2004=dict(online=663.0, offline=0.74),
        NP2004=dict(online=653.2, offline=0.717),
        TD2004=dict(online=263.3, offline=0.314),
        MQ2007=dict(online=385.9, offline=0.356),
        MQ2008=dict(online=501.5, offline=0.468),
        OHSUMED=dict(online=482.6, offline=0.439)
    ),
    inf=dict(
        HP2003=dict(online=650.9, offline=0.759),
        NP2003=dict(online=603.0, offline=0.704),
        TD2003=dict(online=251.6, offline=0.286),
        HP2004=dict(online=616.1, offline=0.732),
        NP2004=dict(online=617.8, offline=0.711),
        TD2004=dict(online=245.0, offline=0.299),
        MQ2007=dict(online=377.2, offline=0.34),
        MQ2008=dict(online=496.3, offline=0.456),
        OHSUMED=dict(online=474.3, offline=0.433)
    )
)

DEFAULT_FILTERS = [
    GroupFilter(column_name='click_model', group='per'),
    GroupFilter(column_name='dataset', group='NP2003')
]


def aggregate_data(path: Path, um: list, datasets: list):
    df = pd.DataFrame()
    for cm in tqdm(um, desc='cm'):
        for ds in tqdm(datasets, desc='dataset'):
            agg = pd.read_csv(path / cm / ds / 'aggregated.csv')
            agg['dataset'] = ds
            agg['click_model'] = cm
            agg['online_benchmark'] = BENCHMARK_DATA[cm][ds]['online']
            agg['offline_benchmark'] = BENCHMARK_DATA[cm][ds]['offline']
            df = df.append(agg)
    df.to_csv(path / 'overall_metrics.csv', header=True, index=True)
    return df


def change_eval_mode(source, p, view):
    return CustomJS(args=dict(source=source, p=p, view=view), code="""
        let data = source.data;
        let f = cb_obj.active;
        let column = f == 0 ? "offline_mean@10" : "online_mean@10";
        
        p.y_range.end = f == 0 ? 1 : 1300;
        data['y'] = data[column];
        source.change.emit();        
    """)


def change_cm(view):
    return CustomJS(args=dict(view=view), code="""
        let cm_filter = view.filters[0];
        let cm_ind = cb_obj.active;
        let cm = "per";        
        if (cm_ind == 1) cm = "nav";
        if (cm_ind == 2) cm = "inf";
        cm_filter.group = cm;
        view.compute_indices();
    """)

def change_ds(view):
    return CustomJS(args=dict(view=view), code="""
            let ds_filter = view.filters[1];
            let ds = cb_obj.value;
            ds_filter.group = ds;
            view.compute_indices();
        """)


def build_dashboard(path: Path, num_queries=1000):
    output_file(str(path / 'dashboard.html'))

    data = pd.read_csv(str(path / 'overall_metrics.csv'), index_col=0, dtype={
        'dataset': str,
        'click_model': str,
        'online_benchmark': np.float64,
        'offline_benchmark': np.float64,
        'online_mean@10': np.float64,
        'online_std@10': np.float64,
        'offline_mean @ 10': np.float64,
        'offline_std @ 10': np.float64
    })
    COLOR_MAPPING = dict(zip(data.dataset.unique(), Category10[9]))

    data['color'] = data.dataset.map(COLOR_MAPPING)

    data['lower_offline'] = data['offline_mean@10'] - data['offline_std@10']
    data['upper_offline'] = data['offline_mean@10'] + data['offline_std@10']

    data['lower_online'] = data['online_mean@10'] - data['online_std@10']
    data['upper_online'] = data['online_mean@10'] - data['online_std@10']

    data['y'] = data['offline_mean@10']
    data['x'] = data.index.values
    source = ColumnDataSource(data)
    view = CDSView(source=source, filters=DEFAULT_FILTERS)

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
    hover = HoverTool(tooltips=[("NDCG@10", "$y")])
    p = figure(tools=TOOLS, x_range=(0, num_queries), y_range=(0, 1))
    p.add_tools(hover)

    p.line(x='x', y='y', source=source, view=view, line_color=data.color.iloc[0], legend='GA-TDM')
    
    p.line(x='x', y='online_benchmark', source=source, view=view, line_color='red', legend='MGD-19')
    p.line(x='x', y='offline_benchmark', source=source, view=view, line_color='black', legend='MGD-19')

    # Widgets
    # Evaluation mode
    eval_mode_rb = RadioButtonGroup(labels=['Offline', 'Online'], active=0, callback=change_eval_mode(source, p, view))
    # Click model
    click_model_rb = RadioButtonGroup(labels=['Perfect', 'Navigational', 'Informational'], active=0, callback=change_cm(view))
    # Dataset
    dataset_select = Select(title="Dataset", value="HP2003", options=list(data.dataset.unique()), callback=change_ds(view))
    
    # Legend
    p.legend.location = 'top_left'

    wb = widgetbox([eval_mode_rb, click_model_rb, dataset_select])

    save(column([wb, p]))


if __name__ == '__main__':
    import argparse
    DATA_PATH = Path('/mnt/c/Users/Rodion_Martynov/Documents/projects/Lerot3/out')
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=False)
    parser.add_argument("-mode", default='dashboard')
    args = parser.parse_args()
    if args.n:
        experiment = args.n
    else:
        raise Exception("Name of experiment is missing")
    
    um = [
        "inf",
        "nav",
        "per"
    ]
    datasets = [
        "OHSUMED",
        "NP2003",
        "HP2003",
        "HP2004",
        "MQ2007",
        "MQ2008",
        "NP2004",
        "TD2003",
        "TD2004"
    ]
    if args.mode == 'aggregate':
        aggregate_data(DATA_PATH / experiment, um, datasets)
    else:
        build_dashboard(DATA_PATH / experiment, num_queries=10000)

    print(f"\\multirow{{9}}{{*}}{{\\begin{{turn}}{{90}}\\textit{{{um[row.Index[0]]}}}\\end{turn}}}")
