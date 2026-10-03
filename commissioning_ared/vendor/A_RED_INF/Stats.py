import matplotlib.pyplot as plt
import numpy as np

"""# Stats"""
class Stats:
    def __init__(self):
        self.kappa_vs_queries = [] # store kappa and querries for a run of ared as follows:
                                   # (kappa, num_queries, num_queries_by_time), kappa and num_queries are ints and num_queries_by_time is a list
        self.kappa_precision_recall = []
        self.averaged_precision_recalls = [] # stored as a list of lists as [kappa, ave precision, ave recall]

    def init_for_kappa_loop(self,kappa):
        self.averaged_precision_recalls.append([kappa, 0, 0, 0, 0])  # kappa, ave precision, ave recall, ave baseline precision, ave baseline recall
        self.precisions = [] # not in constructor because resetting every kappa loop iteration
        self.recalls = []
        self.precision_baseline = []
        self.recall_baseline = []

    def store_ared_query_information(self, ared):
        num_queries = len(ared.l_buf.abs_idx_array)
        num_queries_by_time = []

        highest_idx = ared.data_window.abs_idx_max

        last_idx = -1
        for i, query_abs_idx in enumerate(ared.l_buf.abs_idx_array):
            diff = query_abs_idx - last_idx
            while 0 < diff:
                num_queries_by_time.append(i + 1)
                diff += -1

            last_idx = query_abs_idx

        diff = highest_idx - last_idx
        i = num_queries_by_time[len(num_queries_by_time) - 1]
        while 0 < diff:
            num_queries_by_time.append(i + 1)
            diff += -1

        self.kappa_vs_queries.append((ared.kappa, num_queries, num_queries_by_time))

    def store_ared_precision_recall(self, kappa, precision, recall, random_precision_baseline, random_recall_baseline):
        self.kappa_precision_recall.append((kappa, precision, recall, random_precision_baseline, random_recall_baseline))

    def graph_queries_over_time(self, kappa, save_path=None, show=True):
        """
        Plots the cumulative number of queries over time for a specific kappa.

        Args:
            kappa (int): The kappa value to plot queries for.
            save_path (str or None): Path to save the figure.
            show (bool): Whether to display the plot.
        """
        # Find the record for the requested kappa
        entry = next((e for e in self.kappa_vs_queries if e[0] == kappa), None)
        if entry is None:
            print(f"No query data found for kappa={kappa}")
            return

        _, _, num_queries_by_time = entry

        plt.figure(figsize=(10, 4))
        plt.plot(num_queries_by_time, color='blue', linewidth=1.5)
        plt.title(f"Cumulative Number of Queries Over Time (kappa={kappa})")
        plt.xlabel("Absolute Data Index")
        plt.ylabel("Cumulative Queries")
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300 if save_path.endswith(".png") else None)

        if show:
            plt.show()
        else:
            plt.close()

    def graph_query_rate_over_time(self, kappa, window_size=100, save_path=None, show=True):
        """
        Plots the query rate over time for a specific kappa.

        Args:
            kappa (int): The kappa value to plot query rate for.
            window_size (int): Size of each time window in data points.
            save_path (str or None): Path to save the figure.
            show (bool): Whether to display the plot.
        """
        entry = next((e for e in self.kappa_vs_queries if e[0] == kappa), None)
        if entry is None:
            print(f"No query data found for kappa={kappa}")
            return

        _, _, num_queries_by_time = entry

        queried_indices = set()
        for i in range(1, len(num_queries_by_time)):
            if num_queries_by_time[i] > num_queries_by_time[i - 1]:
                queried_indices.add(i)
        if num_queries_by_time and num_queries_by_time[0] > 0:
            queried_indices.add(0)

        highest_idx = len(num_queries_by_time)
        num_windows = (highest_idx + window_size - 1) // window_size

        rates = []
        xs = []

        for w in range(num_windows):
            start = w * window_size
            end = min((w + 1) * window_size, highest_idx)
            window_indices = range(start, end)
            num_queried = sum(1 for idx in window_indices if idx in queried_indices)
            rate = (num_queried / (end - start)) * 100  # percentage
            rates.append(rate)
            xs.append(start)

        plt.figure(figsize=(10, 4))
        plt.bar(xs, rates, width=window_size * 0.8, color='mediumseagreen', edgecolor='black')
        plt.title(f"Query Rate Over Time (kappa={kappa}, window={window_size})")
        plt.xlabel("Absolute Data Index")
        plt.ylabel("Query Rate (%)")

        max_rate = max(rates) if rates else 1
        plt.ylim(0, max_rate * 1.1)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300 if save_path.endswith(".png") else None)

        if show:
            plt.show()
        else:
            plt.close()

    def graph_all_queries_over_time(self, save_path=None, show=True):
        """
        Plots cumulative queries over time for all stored kappa values on a single graph.

        Args:
            save_path (str or None): Path to save the figure.
            show (bool): Whether to display the plot.
        """
        plt.figure(figsize=(12, 6))

        if not self.kappa_vs_queries:
            print("No query data available to plot.")
            return

        for kappa, _, num_queries_by_time in self.kappa_vs_queries:
            plt.plot(num_queries_by_time, label=f'kappa={kappa}', linewidth=1.5)

        plt.title("Cumulative Number of Queries Over Time (All Kappas)")
        plt.xlabel("Absolute Data Index")
        plt.ylabel("Cumulative Queries")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300 if save_path.endswith(".png") else None)

        if show:
            plt.show()
        else:
            plt.close()

    def graph_all_query_rates_over_time(self, window_size=100, save_path=None, show=True):
        """
        Plots query rates over time for all stored kappa values on a single graph.

        Args:
            window_size (int): Size of each time window in data points.
            save_path (str or None): Path to save the figure.
            show (bool): Whether to display the plot.
        """
        plt.figure(figsize=(12, 6))

        if not self.kappa_vs_queries:
            print("No query data available to plot.")
            return

        for kappa, _, num_queries_by_time in self.kappa_vs_queries:
            queried_indices = set()
            for i in range(1, len(num_queries_by_time)):
                if num_queries_by_time[i] > num_queries_by_time[i - 1]:
                    queried_indices.add(i)
            if num_queries_by_time and num_queries_by_time[0] > 0:
                queried_indices.add(0)

            highest_idx = len(num_queries_by_time)
            num_windows = (highest_idx + window_size - 1) // window_size

            rates = []
            xs = []

            for w in range(num_windows):
                start = w * window_size
                end = min((w + 1) * window_size, highest_idx)
                window_indices = range(start, end)
                num_queried = sum(1 for idx in window_indices if idx in queried_indices)
                rate = (num_queried / (end - start)) * 100  # percentage
                rates.append(rate)
                xs.append(start)

            plt.plot(xs, rates, label=f'kappa={kappa}', linewidth=1.5)

        plt.title(f"Query Rate Over Time (Window Size = {window_size})")
        plt.xlabel("Absolute Data Index")
        plt.ylabel("Query Rate (%)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300 if save_path.endswith(".png") else None)

        if show:
            plt.show()
        else:
            plt.close()

    def plot_precision_recall_curve(self, save_path=None, show=True):
        if not self.kappa_precision_recall:
            print("No precision-recall data to plot.")
            return

        # Sort by kappa for consistent plotting           Stats()
        sorted_data = sorted(self.kappa_precision_recall, key=lambda x: x[0])
        kappas = [x[0] for x in sorted_data]
        precisions = [x[1] for x in sorted_data]
        recalls = [x[2] for x in sorted_data]
        rand_precisions = [x[3] for x in sorted_data]
        rand_recalls = [x[4] for x in sorted_data]

        print(kappas, precisions, recalls, rand_precisions, rand_recalls)

        plt.figure(figsize=(8, 6))

        # Plot ARED points
        plt.scatter(recalls, precisions, color='blue', marker='o', label='ARED')

        # Plot random baseline points
        plt.scatter(rand_recalls, rand_precisions, color='red', marker='x', label='Random baseline')

        # Add labels for ARED points
        for kappa, r, p in zip(kappas, recalls, precisions):
            plt.annotate(f'κ={kappa}',
                         xy=(r, p),
                         xytext=(5, 5),  # offset in points
                         textcoords='offset points',
                         fontsize=10,
                         color='blue',
                         arrowprops=dict(arrowstyle='-', lw=0.5, color='blue'))

        # Add labels for random baseline points
        for kappa, r, p in zip(kappas, rand_recalls, rand_precisions):
            plt.annotate(f'κ={kappa}',
                         xy=(r, p),
                         xytext=(-25, -10),  # offset in points
                         textcoords='offset points',
                         fontsize=10,
                         color='red',
                         arrowprops=dict(arrowstyle='-', lw=0.5))

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall (ARED vs Random Baseline)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300 if save_path.endswith(".png") else None)

        if show:
            plt.show()
        else:
            plt.close()