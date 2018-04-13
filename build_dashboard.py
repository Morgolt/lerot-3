from pathlib import Path

import pandas as pd
from bokeh.io import output_file, save
from bokeh.layouts import widgetbox, column
from bokeh.models import ColumnDataSource, CDSView, CustomJS, GroupFilter, HoverTool
from bokeh.models.widgets import RadioButtonGroup, Select
from bokeh.palettes import Category10
from bokeh.plotting import figure
from tqdm import tqdm

COLUMN_MAPPING = dict(
    online='online_mean@10',
    online_std='online_std@10',
    offline='offline_mean@10',
    offline_std='offline_std@10'
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
            df = df.append(agg)
    df.to_csv(path / 'overall_metrics.csv', header=True, index=True)
    return df


def change_eval_mode(source, p, view):
    return CustomJS(args=dict(source=source, p=p, view=view), code="""
        let data = source.data;
        let f = cb_obj.active;
        let column = f == 0 ? "offline_mean@10" : "online_mean@10";
        
        p.y_range.end = f == 0 ? 1 : 200;
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


def build_dashboard(path: Path):
    output_file(str(path / 'dashboard.html'))

    data = pd.read_csv(str(path / 'overall_metrics.csv'), index_col=0)
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
    p = figure(tools=TOOLS, x_range=(0, 1000), y_range=(0, 1))
    p.add_tools(hover)

    p.line(x='x', y='y', source=source, view=view, line_color=data.color.iloc[0])

    # Widgets
    # Evaluation mode
    eval_mode_rb = RadioButtonGroup(labels=['Offline', 'Online'], active=0, callback=change_eval_mode(source, p, view))
    # Click model
    click_model_rb = RadioButtonGroup(labels=['Perfect', 'Navigational', 'Informational'], active=0, callback=change_cm(view))
    # Dataset
    dataset_select = Select(title="Dataset", value="HP2003", options=list(data.dataset.unique()), callback=change_ds(view))

    wb = widgetbox([eval_mode_rb, click_model_rb, dataset_select])

    save(column([wb, p]))


if __name__ == '__main__':
    DATA_PATH = Path('D:/Projects/Lerot3/out')
    experiment = 'ga_baseline_mt'
    # um = [
    #     "inf",
    #     "nav",
    #     "per"
    # ]
    # datasets = [
    #     "OHSUMED",
    #     "NP2003",
    #     "HP2003",
    #     "HP2004",
    #     "MQ2007",
    #     "MQ2008",
    #     "NP2004",
    #     "TD2003",
    #     "TD2004"
    # ]
    # aggregate_data(DATA_PATH / experiment, um, datasets)
    build_dashboard(DATA_PATH / experiment)
