import random
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, show
from deap import creator, base, tools
from tqdm import tqdm

from lerot.evaluation import NdcgEval
from lerot.interleave import ProbabilisticMultileave, TeamDraftMultileave
from lerot.query import load_queries
from lerot.ranker.GARankingFunction import GARankingFunction
from lerot.user import CascadeUserModel
from lerot.utils import sample_unit_sphere

QUERY_COUNT = 0


def _sample_qid(query_keys, query_count, query_length):
    return query_keys[np.random.randint(0, query_length - 1)]


def get_ranked_list(query, population, comparison):
    rankers = init_ranking(population)
    (l, context) = comparison.interleave(rankers, query, 10)
    return l, context


def init_ranking(population):
    rankers = []
    for ind in population:
        rankers.append(GARankingFunction(ind))
    return rankers


def update_solution(rl, best, ctx, c, population, comparison, q):
    creds = comparison.infer_outcome(rl, ctx, c, q)
    # todo: idea - use gained values as DCG relevance labels -> DCG Fitness
    for part, fit in zip(population, creds):
        part.fitness.values = (fit,)
        if part.best is None or part.best.fitness < part.fitness:
            part.best = creator.Particle(part)
            part.best.fitness.values = part.fitness.values
        if best is None or best.fitness < part.fitness:
            best = creator.Particle(part)
            best.fitness.values = part.fitness.values
            # print(best.fitness.values[0])
    for part in population:
        toolbox.update(part, best)

    return best, population


def plot_metrics(df, outdir):
    p = figure(plot_width=1200, plot_height=600)
    p.title.text = 'Metrics on current run'
    for c in df.columns:
        p.line(df.index, df[c], line_width=2, alpha=0.8, muted_alpha=0.2, legend=c)
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    output_file(str(outdir / "interactive_legend.html"), title="Metrics")

    show(p)


def save_results(offline_evaluation, online_evaluation, out: Path, plot=False):
    online_df = pd.DataFrame.from_dict(online_evaluation, dtype=np.float64)
    offline_df = pd.DataFrame.from_dict(offline_evaluation, dtype=np.float64)
    df = online_df.join(offline_df, lsuffix='_online', rsuffix='_offline')
    if not out.exists():
        out.mkdir(parents=True)

    i = 0
    while (out / f"{i:03d}").exists():
        i += 1
    (out / f"{i:03d}").mkdir()
    outdir = (out / f"{i:03d}")

    df.to_csv(str(outdir / 'metrics.csv'), index=False, header=True, float_format="%.5f")
    if plot:
        plot_metrics(df, outdir)


def init_particle(size, pmin, pmax, smin, smax):
    part = creator.Particle(sample_unit_sphere(size))
    part.speed = sample_unit_sphere(size) * GAMMA
    part.smin = smin
    part.smax = smax
    return part


def update_particle(part, best, phi1, phi2):
    u1 = phi1 * np.random.rand(len(part))
    u2 = phi2 * np.random.rand(len(part))
    v_u1 = u1 * (part.best - part)
    v_u2 = u2 * (best - part)
    part.speed += v_u1 + v_u2
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part += ALPHA * part.speed


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    feature_count = 45
    num_rankers = 20
    num_queries = 1000
    GAMMA = 1
    ALPHA = 0.5

    creator.create("FitnessClicks", base.Fitness, weights=(1.0,))
    creator.create("Particle", np.ndarray, fitness=creator.FitnessClicks, speed=list, smin=None, smax=None, best=None)

    toolbox = base.Toolbox()
    toolbox.register("particle", init_particle, size=feature_count, pmin=-1, pmax=1, smin=-GAMMA,
                     smax=GAMMA)
    toolbox.register("update", update_particle, phi1=ALPHA, phi2=ALPHA)

    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", update_particle, phi1=0, phi2=0)

    multileave = TeamDraftMultileave()
    # ProbabilisticMultileave()

    um = CascadeUserModel("--p_click 0:0.0, 1:0.5, 2:1.0 --p_stop  0:0, 1:0, 2:0")

    pop = toolbox.population(n=num_rankers)

    fn = Path("D:/Projects/Lerot3/data/OHSUMED/Fold1")
    train = load_queries(str(fn / "train.txt"), 45, False)
    test = load_queries(str(fn / "test.txt"), 45, False)
    out = Path("D:/Projects/Lerot3/out/pso_test")

    query_keys = sorted(train.keys())
    query_length = len(query_keys)

    online_evaluation = {}
    offline_evaluation = {}

    evaluations = [('evaluation.NdcgEval', dict(cutoff=1, eval_class=NdcgEval())),
                   ('evaluation.NdcgEval', dict(cutoff=3, eval_class=NdcgEval())),
                   ('evaluation.NdcgEval', dict(cutoff=5, eval_class=NdcgEval())),
                   ('evaluation.NdcgEval', dict(cutoff=7, eval_class=NdcgEval())),
                   ('evaluation.NdcgEval', dict(cutoff=10, eval_class=NdcgEval()))]
    for eval_name, eval_dict in evaluations:
        dict_name = eval_name + '@' + str(eval_dict['cutoff'])
        # Stop if there are duplicate evaluations
        online_evaluation[dict_name] = []
        offline_evaluation[dict_name] = []
    similarities = [.0]

    current_best_w = pop[np.random.randint(0, num_rankers)]
    best = None
    for query_count in tqdm(range(num_queries)):
        previous_solution_w = current_best_w
        qid = _sample_qid(query_keys, query_count, query_length)
        query = train[qid]
        result_list, context = get_ranked_list(query, pop, multileave)

        for eval_name, eval_dict in evaluations:
            a = float(eval_dict['eval_class'].evaluate_ranking(result_list, query, eval_dict['cutoff']))
            online_evaluation[eval_name + '@' + str(eval_dict['cutoff'])].append(a)

        clicks = um.get_clicks(result_list, query.get_labels())

        current_solution, pop = update_solution(result_list, best, context, clicks, pop, multileave, query)

        for eval_name, eval_dict in evaluations:
            # Create dict name as done above
            dict_name = eval_name + '@' + str(eval_dict['cutoff'])
            e1 = eval_dict['eval_class'].evaluate_all(GARankingFunction(current_solution), test, eval_dict['cutoff'])
            offline_evaluation[dict_name].append(float(e1))

            # print("Current offline %s = %.3f" % (dict_name, offline_evaluation[dict_name][-1]))
        # print(f"Query #{query_count} done")

    save_results(offline_evaluation, online_evaluation, out, plot=True)
