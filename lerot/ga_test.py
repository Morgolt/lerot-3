import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.layouts import row, column
from bokeh.plotting import figure, output_file, show
from deap import creator, base, tools
from scoop import futures
from tqdm import tqdm

from lerot.evaluation import NdcgEval
from lerot.interleave import ProbabilisticMultileave, TeamDraftMultileave
from lerot.query import load_queries
from lerot.ranker.GARankingFunction import GARankingFunction
from lerot.user import CascadeUserModel
from lerot.utils import sample_unit_sphere
from bokeh.palettes import Set1_5

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


def update_solution(rl, ctx, c, population, comparison, q):
    creds = comparison.infer_outcome(rl, ctx, c, q)
    # todo: idea - use gained values as DCG relevance labels -> DCG Fitness
    if creds is not None:
        for ind, fit in zip(population, creds):
            ind.fitness.values = (fit,)

    current_best_w = copy.deepcopy(tools.selBest(population, 1))[0]


    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    population[:] = offspring
    return current_best_w, population


def plot_metrics(offline_df, online_df, outdir, fitness=None):
    off = figure(plot_width=1200, plot_height=400, x_range=(0, len(offline_df)), y_range=(0, 1))
    off.title.text = 'Offline NDCG'
    for c, color in zip(offline_df.columns, Set1_5):
        off.line(offline_df.index, offline_df[c], line_width=2, alpha=0.8, muted_alpha=0.2, legend=c, color=color)
    off.legend.location = "top_left"
    off.legend.click_policy = "hide"

    for c in online_df.columns:
        print(f"Final online {c}: {online_df[c].sum()}")
    
    # on = figure(plot_width=1200, plot_height=600, x_range=(0, len(offline_df)), y_range=(0, online_df.values.max()))
    # on.title.text = 'online NDCG'
    # for c, color in zip(online_df.columns, Set1_5):
    #     on.line(online_df.index, online_df[c], line_width=2, alpha=0.8, muted_alpha=0.2, legend=c, color=color)
    # on.legend.location = "top_left"
    # on.legend.click_policy = "hide"

    output_file(str(outdir / "interactive_legend.html"), title="Metrics")

    show(off)


def save_results(offline_evaluation, online_evaluation, out: Path, fitness=None, plot=False, gamma=0.995):
    online_df = pd.DataFrame.from_dict(online_evaluation, dtype=np.float64)
    gm = online_df.index.map(lambda x: gamma ** x)
    online_df = online_df.multiply(gm, axis='index')
    offline_df = pd.DataFrame.from_dict(offline_evaluation, dtype=np.float64)
    df = online_df.join(offline_df, lsuffix='_online', rsuffix='_offline')
    if fitness:
        df['fitness'] = pd.Series(data=fitness, index=online_df.index)
    if not out.exists():
        out.mkdir(parents=True)

    i = 0
    while (out / f"{i:03d}").exists():
        i += 1
    (out / f"{i:03d}").mkdir()
    outdir = (out / f"{i:03d}")

    df.to_csv(str(outdir / 'metrics.csv'), index=False, header=True)
    if plot:
        plot_metrics(offline_df, online_df, outdir)


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    feature_count = 64
    num_rankers = 19
    num_queries = 10000
    CXPB = 0.9
    MUTPB = 0.5
    discount = 0.995

    creator.create("FitnessClicks", base.Fitness, weights=(1.0,))

    creator.create("Individual", list, fitness=creator.FitnessClicks)

    toolbox = base.Toolbox()
    toolbox.register("map", futures.map)
    toolbox.register("attr_float", sample_unit_sphere, feature_count)
    toolbox.register("ind", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.05, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    toolbox.register("population", tools.initRepeat, list, toolbox.ind)

    multileave = TeamDraftMultileave()

    um = CascadeUserModel("--p_click 0:0, 1:0.5, 2:1 --p_stop  0:0, 1:0, 2:0")

    pop = toolbox.population(n=num_rankers)

    dataset = "NP2003"
    fold = "Fold2"
    fn = Path("D:/Projects/Lerot3/data/") / dataset / fold
    train = load_queries(str(fn / "train.txt"), feature_count, False)
    test = load_queries(str(fn / "test.txt"), feature_count, False)
    out = Path("D:/Projects/Lerot3/out/ga_test") / dataset / fold

    query_keys = sorted(train.keys())
    query_length = len(query_keys)

    online_evaluation = {}
    offline_evaluation = {}

    evaluations = [('Ndcg', dict(cutoff=1, eval_class=NdcgEval())),
                   ('Ndcg', dict(cutoff=3, eval_class=NdcgEval())),
                   ('Ndcg', dict(cutoff=5, eval_class=NdcgEval())),
                   ('Ndcg', dict(cutoff=7, eval_class=NdcgEval())),
                   ('Ndcg', dict(cutoff=10, eval_class=NdcgEval()))]
    fitness = []
    for eval_name, eval_dict in evaluations:
        dict_name = eval_name + '@' + str(eval_dict['cutoff'])
        # Stop if there are duplicate evaluations
        online_evaluation[dict_name] = []
        offline_evaluation[dict_name] = []
    similarities = [.0]

    current_best_w = pop[np.random.randint(0, num_rankers)]
    for query_count in tqdm(range(num_queries)):
        previous_solution_w = current_best_w
        qid = _sample_qid(query_keys, query_count, query_length)
        query = train[qid]
        result_list, context = get_ranked_list(query, pop, multileave)

        for eval_name, eval_dict in evaluations:
            a = float(eval_dict['eval_class'].evaluate_ranking(result_list, query, eval_dict['cutoff']))
            online_evaluation[eval_name + '@' + str(eval_dict['cutoff'])].append(a)

        clicks = um.get_clicks(result_list, query.get_labels())

        current_solution, pop = update_solution(result_list, context, clicks, pop, multileave, query)

        fitness.append(current_solution.fitness.values[0])

        for eval_name, eval_dict in evaluations:
            # Create dict name as done above
            dict_name = eval_name + '@' + str(eval_dict['cutoff'])
            e1 = eval_dict['eval_class'].evaluate_all(GARankingFunction(current_solution), test, eval_dict['cutoff'])
            offline_evaluation[dict_name].append(float(e1))

    save_results(offline_evaluation, online_evaluation, out, fitness=fitness, plot=True)
