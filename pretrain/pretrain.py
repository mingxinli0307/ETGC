import argparse
import os
from pathlib import Path

import networkx as nx

import node2vec
from transform_data_to_edgelist import trans_data_to_edge


HINOS_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = HINOS_DIR / "dataset"
PRETRAIN_DIR = HINOS_DIR / "pretrain"


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec pretraining for HiNoS.")
    parser.add_argument("--data", default="school", help="Dataset name.")
    parser.add_argument(
        "--input",
        default=None,
        help="Input edgelist file. Defaults to HiNoS/dataset/<data>/<data>.edgelist.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output embedding file. Defaults to HiNoS/pretrain/<data>_feature.emb.",
    )
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--walk-length", type=int, default=80)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--iter", default=1, type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)

    parser.add_argument("--weighted", dest="weighted", action="store_true")
    parser.add_argument("--unweighted", dest="weighted", action="store_false")
    parser.set_defaults(weighted=False)

    parser.add_argument("--directed", dest="directed", action="store_true")
    parser.add_argument("--undirected", dest="directed", action="store_false")
    parser.set_defaults(directed=False)
    return parser.parse_args()


def read_graph(args):
    if args.weighted:
        graph = nx.read_edgelist(
            args.input,
            nodetype=int,
            data=(("weight", float),),
            create_using=nx.DiGraph(),
        )
    else:
        graph = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in graph.edges():
            graph[edge[0]][edge[1]]["weight"] = 1.0

    if not args.directed:
        graph = graph.to_undirected()
    return graph


def learn_embeddings(walks, args):
    from gensim.models import Word2Vec

    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(
        walks,
        vector_size=args.dimensions,
        window=args.window_size,
        min_count=0,
        sg=1,
        workers=args.workers,
        epochs=args.iter,
    )
    model.wv.save_word2vec_format(args.output)


def main(args):
    PRETRAIN_DIR.mkdir(parents=True, exist_ok=True)

    if args.input is None:
        args.input = str(DATASET_DIR / args.data / f"{args.data}.edgelist")
    if args.output is None:
        args.output = str(PRETRAIN_DIR / f"{args.data}_feature.emb")

    if not os.path.exists(args.input):
        txt_path = DATASET_DIR / args.data / f"{args.data}.txt"
        if not txt_path.exists():
            raise FileNotFoundError(f"Missing temporal edge file: {txt_path}")
        trans_data_to_edge(str(txt_path), args.input)

    print(f"[Pretrain] input  = {args.input}")
    print(f"[Pretrain] output = {args.output}")

    nx_graph = read_graph(args)
    graph = node2vec.Graph(nx_graph, args.directed, args.p, args.q)
    graph.preprocess_transition_probs()
    walks = graph.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks, args)
    print("[Pretrain] finished.")


if __name__ == "__main__":
    main(parse_args())
