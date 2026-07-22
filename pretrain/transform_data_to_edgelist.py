from pathlib import Path


def trans_data_to_edge(data_path: str, edge_path: str) -> None:
    data_path = Path(data_path)
    edge_path = Path(edge_path)
    edge_path.parent.mkdir(parents=True, exist_ok=True)

    with data_path.open("r", encoding="utf-8") as infile, edge_path.open("w", encoding="utf-8") as out:
        for line in infile:
            parts = line.split()
            if len(parts) < 2:
                continue
            out.write(f"{parts[0]} {parts[1]}\n")

    print(f"Edgelist transform finished: {edge_path}")
