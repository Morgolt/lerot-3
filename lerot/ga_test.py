import argparse
import logging
import copy
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
from bokeh.palettes import Set1_5
from bokeh.plotting import figure, output_file
from deap import creator, base, tools
from joblib import Parallel, delayed

from lerot.evaluation import NdcgEval
from lerot.interleave import TeamDraftMultileave, ProbabilisticMultileave
from lerot.query import load_queries
from lerot.ranker.GARankingFunction import GARankingFunction
from lerot.user import CascadeUserModel
from lerot.utils import sample_unit_sphere

RUN_CONFIG = namedtuple(
    'RUN_CONFIG',
    [
        'num_rankers', 'num_queries',
        'CXPB', 'MUTPB',
        'in_path', 'out_path',
        'cm', 'ds', 'fold',
        'interleave'
    ]
)


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


def update_solution(rl, ctx, c, population, comparison, q,
                    logger, toolbox, hof, stats, config: RUN_CONFIG):
    creds = comparison.infer_outcome(rl, ctx, c, q)
    # todo: idea - use gained values as DCG relevance labels -> DCG Fitness
    if creds is not None:
        for ind, fit in zip(population, creds):
            ind.fitness.values = (fit,)
        logger.record(**stats.compile(population))
        
    ##### Baseline selection - current best
    current_best_w = copy.deepcopy(tools.selBest(population, 1))[0]

    ##### Average of whole population: -0.05 ndcg
    # current_best_w = np.mean(population, axis=0)

    ##### Average hall of fame: -0.4
    # current_best_w = np.mean(hof.items, axis=1)

    ##### Moving average
    # current_best_w =

    if len(hof.items) < hof.maxsize:
        bst = list(map(toolbox.clone, tools.selBest(population, 1)))
        for ind in bst:
            hof.insert(ind)
    else:
        hof.update(population)

    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if np.random.rand() < config.CXPB:
            # todo: reevaluate individuals based on the same clicks
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if np.random.rand() < config.MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    population[:] = toolbox.select(offspring + hof.items, len(population))
    
    return current_best_w, population


def save_results(offline_evaluation, online_evaluation, out: Path, fitness=None, plot=False, gamma=0.995):
    online_df = pd.DataFrame.from_dict(online_evaluation, dtype=np.float64)
    # gm = online_df.index.map(lambda x: gamma ** x)
    # online_df = online_df.multiply(gm, axis='index')
    offline_df = pd.DataFrame.from_dict(offline_evaluation, dtype=np.float64)
    df = online_df.join(offline_df, lsuffix='_online', rsuffix='_offline')
    if fitness:
        df['fitness'] = pd.Series(data=fitness, index=online_df.index)

    df.to_csv(str(out / 'metrics.csv'), index=False, header=True)

        
def filter_config(config: RUN_CONFIG):
    if (config.out_path / 'metrics.csv').exists():
        logging.warning(f"Run cm: {config.cm['name']}, ds: {config.ds['name']}, fold: {config.fold}, run: {config.out_path.parts[-1]} already finished, skipping.")
        return False
    else:
        return True


def run(config: RUN_CONFIG, train, test):
    logging.warning(
        f"Started processing cm: {config.cm['name']}, ds: {config.ds['name']}, fold: {config.fold}, run: {config.out_path.parts[-1]}.")
    feature_count = config.ds["feature_count"]
    # fn = config.in_path
    # train = load_queries(str(fn / "train.txt"), feature_count, False)
    # test = load_queries(str(fn / "test.txt"), feature_count, False)
    out = config.out_path
    if not out.exists():
        out.mkdir(parents=True)

    if "FitnessClicks" not in creator.__dict__:
        creator.create("FitnessClicks", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessClicks)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", sample_unit_sphere, feature_count)
    toolbox.register("ind", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logger = tools.Logbook()

    toolbox.register("population", tools.initRepeat, list, toolbox.ind)
    halloffame = tools.HallOfFame(1)
    pop = toolbox.population(n=config.num_rankers)
    #######################################################################
    multileave = TeamDraftMultileave() if config.interleave == 'tdm' else ProbabilisticMultileave()
    #######################################################################

    query_keys = sorted(train.keys())
    query_length = len(query_keys)

    online_evaluation = {}
    offline_evaluation = {}

    evaluations = [
        # ('Ndcg', dict(cutoff=1, eval_class=NdcgEval())),
        # ('Ndcg', dict(cutoff=3, eval_class=NdcgEval())),
        # ('Ndcg', dict(cutoff=5, eval_class=NdcgEval())),
        # ('Ndcg', dict(cutoff=7, eval_class=NdcgEval())),
        ('Ndcg', dict(cutoff=10, eval_class=NdcgEval()))
    ]
    for eval_name, eval_dict in evaluations:
        dict_name = eval_name + '@' + str(eval_dict['cutoff'])
        online_evaluation[dict_name] = []
        offline_evaluation[dict_name] = []
        
    cm = config.cm[config.ds['cm']]

    # past_best = []
    # with open(out / 'fitness.log', 'a+') as lf:
    for query_count in range(config.num_queries):
        qid = _sample_qid(query_keys, query_count, query_length)
        query = train[qid]
        result_list, context = get_ranked_list(query, pop, multileave)

        for eval_name, eval_dict in evaluations:
            a = float(eval_dict['eval_class'].evaluate_ranking(result_list, query, eval_dict['cutoff']))
            online_evaluation[eval_name + '@' + str(eval_dict['cutoff'])].append(a)

        clicks = cm.get_clicks(result_list, query.get_labels())

        current_solution, pop = update_solution(result_list, context, clicks, pop, multileave, query,
                                                logger, toolbox, halloffame, stats, config)

        for eval_name, eval_dict in evaluations:
            # Create dict name as done above
            dict_name = eval_name + '@' + str(eval_dict['cutoff'])
            e1 = eval_dict['eval_class'].evaluate_all(GARankingFunction(current_solution), test,
                                                      eval_dict['cutoff'])
            offline_evaluation[dict_name].append(float(e1))
        # if logger:
        #     lf.write(logger.stream)
        #     lf.write('\n')
    with open(out / 'fitness.log', 'a+') as lf:
        lf.write(logger.stream)
    save_results(offline_evaluation, online_evaluation, out)


def main():
    # np.random.seed(42)
    # random.seed(42)
    # feat_count = 64
    # todo: remove unused features
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", default=False)
    parser.add_argument("-n", default=False)
    args = parser.parse_args()

    num_rankers = 19
    num_queries = 10000
    CXPB = 0.5
    MUTPB = 0.3
    num_runs = 25
    interleave = 'tdm'

    verbose = 49
    n_jobs = 5
    # ${optimizer}_${multileaving}_${num-rankers}_${crossover-prob}_${mutation-prob}
    if not args.n:
        raise Exception("Enter name of experiment")
    
    exp_name = args.n
    # base_in_path = Path("C:/Users/Rodion_Martynov/Documents/projects/Lerot3/data")
    # base_out_path = Path("C:/Users/Rodion_Martynov/Documents/projects/Lerot3/out") / exp_name

    base_in_path = Path("/mnt/c/Users/Rodion_Martynov/Documents/projects/Lerot3/data/")
    base_out_path = Path("/mnt/c/Users/Rodion_Martynov/Documents/projects/Lerot3/out") / exp_name

    if base_out_path.exists() and not args.c:
        raise Exception("EXPERIMENT ALREADY CONDUCTED")

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)

    um = [
        dict(
            name="per", 
            cm2=CascadeUserModel("--p_click 0:0, 1:1, 2:1 --p_stop 0:0, 1:0, 2:0"),
            cm3=CascadeUserModel("--p_click 0:0, 1:0.5, 2:1 --p_stop 0:0, 1:0, 2:0")            
        ),
        dict(
            name="nav", 
            cm2=CascadeUserModel("--p_click 0:0.05, 1:0.95, 2:0.95 --p_stop 0:0.2, 1:0.9, 2:0.9"),
            cm3=CascadeUserModel("--p_click 0:0.05, 1:0.5, 2:0.95 --p_stop  0:0.2, 1:0.5, 2:0.9")),
        dict(
            name="inf", 
            cm2=CascadeUserModel("--p_click 0:0.4, 1:0.9, 2:0.9 --p_stop 0:0.1, 1:0.5, 2:0.5"),
            cm3=CascadeUserModel("--p_click 0:0.4, 1:0.7, 2:0.9 --p_stop  0:0.1, 1:0.3, 2:0.5")
        ),
    ]

    ds = [
        dict(name="NP2004", feature_count=64, cm='cm2'),
        dict(name="HP2003", feature_count=64, cm='cm2'),
        dict(name="HP2004", feature_count=64, cm='cm2'),
        dict(name="MQ2007", feature_count=46, cm='cm3'),
        dict(name="MQ2008", feature_count=46, cm='cm3'),
        dict(name="TD2003", feature_count=64, cm='cm2'),
        dict(name="TD2004", feature_count=64, cm='cm2'),
        dict(name="OHSUMED", feature_count=45, cm='cm3'),
        dict(name="NP2003", feature_count=64, cm='cm2'),
    ]
    folds = ["Fold" + str(i) for i in range(1, 6)]
    with Parallel(n_jobs=n_jobs, verbose=verbose) as parallel:
        for dataset in ds:
            for fold in folds:
                fn = base_in_path / dataset["name"] / fold
                feature_count = dataset['feature_count']
                train = load_queries(str(fn / "train.txt"), feature_count, False)
                test = load_queries(str(fn / "test.txt"), feature_count, False)
                for cm in um:
                    configs = [RUN_CONFIG(num_rankers=num_rankers,
                                          num_queries=num_queries,
                                          CXPB=CXPB,
                                          MUTPB=MUTPB,
                                          in_path=base_in_path / dataset["name"] / fold,
                                          out_path=base_out_path / cm["name"] / dataset["name"] / fold / f"{i:03d}",
                                          cm=cm,
                                          ds=dataset,
                                          fold=fold,
                                          interleave=interleave) for i in range(0, num_runs)]
                    configs = filter(filter_config, configs)
                    parallel(delayed(run)(config, train, test) for config in configs)
                del train
                del test


if __name__ == '__main__':
    main()
