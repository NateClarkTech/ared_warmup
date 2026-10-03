import json
from collections import defaultdict

def find_best_settings(results_file):
    """Load results.json and find the best setting for each dataset based on the product of single_rel_recall and avg_query_precision."""
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Results file {results_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {results_file}.")
        return

    # Group results by DATA_SOURCE
    groups = defaultdict(list)
    for result in results:
        data_source = result.get('config', {}).get('DATA_SOURCE')
        if data_source:
            groups[data_source].append(result)

    # For each dataset, find the config with the highest product
    for data_source, res_list in groups.items():
        if not res_list:
            continue

        best_result = max(res_list, key=lambda r: r.get('single_rel_recall', 0) * r.get('avg_query_precision', 0))
        product = best_result['single_rel_recall'] * best_result['avg_query_precision']

        print(f"\nBest setting for dataset: {data_source}")
        print(f"Product (single_rel_recall * avg_query_precision): {product:.4f}")
        print("Configuration:")
        for key, value in best_result['config'].items():
            print(f"  {key}: {value}")

if __name__ == '__main__':
    results_file = "config_files/results.json"
    find_best_settings(results_file)