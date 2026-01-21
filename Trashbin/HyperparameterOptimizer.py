import itertools
import torch

from utils.TrainingUtils import TrainingUtils
from models.GCN import GCN


class HyperparameterOptimizer:
    def __init__(self, seed: int = 0):
        self.seed = seed

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)

    def _normalize_focus_keys(self, focus):
        focus_map = {
            "lr": "lr",
            "learning_rate": "lr",
            "hidden": "hidden",
            "hidden_size": "hidden",
            "dropout": "dropout",
            "weight_decay": "weight_decay",
            "wd": "weight_decay",
        }
        if str(focus).lower() == "all":
            focus_keys = ["lr", "hidden", "dropout", "weight_decay"]
        elif isinstance(focus, (list, tuple, set)):
            focus_keys = [focus_map.get(str(f).lower(), None) for f in focus]
        else:
            focus_keys = [focus_map.get(str(focus).lower(), None)]

        if any(k is None for k in focus_keys):
            raise ValueError("focus must be one of: lr, hidden, dropout, weight_decay")
        return focus_keys

    def _plot_gcn_runs(self, runs, *, focus, epochs: int, show: bool = True, ax=None):
        focus_keys = self._normalize_focus_keys(focus)

        import matplotlib.pyplot as plt

        if len(focus_keys) == 1:
            if ax is None:
                _, ax = plt.subplots()
            axes = [ax]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            axes = list(axes.flatten())

        epochs_axis = list(range(1, epochs + 1))
        for i, focus_key in enumerate(focus_keys):
            if i >= len(axes):
                break
            ax_i = axes[i]
            histories = self._aggregate_runs(runs, focus_key)
            for value in sorted(histories.keys(), key=lambda x: float(x)):
                curves = histories[value]
                if not curves:
                    continue
                mean_curve = torch.tensor(curves, dtype=torch.float).mean(dim=0).tolist()
                ax_i.plot(epochs_axis, mean_curve, label=f"{focus_key}={value}")

            ax_i.set_xlabel("epoch")
            ax_i.set_ylabel("test accuracy")
            ax_i.set_title(f"Test accuracy by epoch vs {focus_key}")
            ax_i.legend()

        if len(focus_keys) > 1:
            for j in range(len(focus_keys), len(axes)):
                axes[j].set_visible(False)

        if show:
            plt.show()

        return axes[0] if len(axes) == 1 else axes

    def _collect_gcn_runs(
        self,
        data,
        *,
        num_classes: int,
        label_attr: str,
        lrs,
        hidden_sizes,
        dropouts,
        weight_decays,
        epochs: int,
        print_every: int,
    ):
        runs = []
        trial = 0
        for lr, hidden, dropout, weight_decay in itertools.product(
            lrs, hidden_sizes, dropouts, weight_decays
        ):
            trial += 1
            self._set_seed(self.seed + trial)

            model = GCN(
                in_channels=data.x.size(1),
                hidden=hidden,
                num_classes=num_classes,
                dropout=dropout,
            )

            _, history = model.fit(
                data,
                label_attr=label_attr,
                lr=lr,
                weight_decay=weight_decay,
                epochs=epochs,
                select_by="test",
                print_every=print_every,
                print_best=False,
                return_history=True,
            )

            runs.append(
                {
                    "lr": lr,
                    "hidden": hidden,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "test": history["test"],
                }
            )

        return runs

    def _aggregate_runs(self, runs, key):
        grouped = {}
        for run in runs:
            grouped.setdefault(run[key], []).append(run["test"])
        return grouped

    def optimize_gcn(
        self,
        data,
        *,
        num_classes: int,
        label_attr: str = "y_task",
        lrs=(1e-3, 3e-3, 1e-2),
        hidden_sizes=(32, 64, 128),
        dropouts=(0.0, 0.2),
        weight_decays=(5e-4,),
        epochs: int = 200,
        print_every: int = 0,
        verbose: bool = True,
        plot: bool = False,
        plot_focus: str = "all",
        plot_show: bool = True,
        plot_ax=None,
        return_runs: bool = False,
    ):
        if not hasattr(data, label_attr):
            raise AttributeError(f"data has no attribute '{label_attr}'")
        y = getattr(data, label_attr)

        best = {
            "test_acc": -1.0,
            "config": None,
            "state_dict": None,
        }

        runs = [] if (plot or return_runs) else None
        trial = 0
        for lr, hidden, dropout, weight_decay in itertools.product(
            lrs, hidden_sizes, dropouts, weight_decays
        ):
            trial += 1
            self._set_seed(self.seed + trial)

            model = GCN(
                in_channels=data.x.size(1),
                hidden=hidden,
                num_classes=num_classes,
                dropout=dropout,
            )

            if runs is None:
                model.fit(
                    data,
                    label_attr=label_attr,
                    lr=lr,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    select_by="test",
                    print_every=print_every,
                    print_best=False,
                )
            else:
                _, history = model.fit(
                    data,
                    label_attr=label_attr,
                    lr=lr,
                    weight_decay=weight_decay,
                    epochs=epochs,
                    select_by="test",
                    print_every=print_every,
                    print_best=False,
                    return_history=True,
                )
                runs.append(
                    {
                        "lr": lr,
                        "hidden": hidden,
                        "dropout": dropout,
                        "weight_decay": weight_decay,
                        "test": history["test"],
                    }
                )

            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                test_acc = TrainingUtils.accuracy(logits, y, data.test_mask)

            if verbose:
                print(
                    f"trial {trial:03d} | lr {lr:g} | hidden {hidden} | "
                    f"dropout {dropout:g} | wd {weight_decay:g} | test {test_acc:.3f}"
                )

            if test_acc > best["test_acc"]:
                best["test_acc"] = test_acc
                best["config"] = {
                    "lr": lr,
                    "hidden": hidden,
                    "dropout": dropout,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                }
                best["state_dict"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best["state_dict"] is None:
            return best

        best_model = GCN(
            in_channels=data.x.size(1),
            hidden=best["config"]["hidden"],
            num_classes=num_classes,
            dropout=best["config"]["dropout"],
        ).to(data.x.device)
        best_model.load_state_dict({k: v.to(data.x.device) for k, v in best["state_dict"].items()})
        best["model"] = best_model
        if runs is not None:
            best["runs"] = runs
            if plot:
                self._plot_gcn_runs(
                    runs,
                    focus=plot_focus,
                    epochs=epochs,
                    show=plot_show,
                    ax=plot_ax,
                )
        return best

    def plot_gcn_hyperparam_effect(
        self,
        data,
        *,
        num_classes: int,
        focus: str,
        label_attr: str = "y_task",
        lrs=(1e-3, 3e-3, 1e-2),
        hidden_sizes=(32, 64, 128),
        dropouts=(0.0, 0.2),
        weight_decays=(5e-4,),
        epochs: int = 200,
        print_every: int = 0,
        show: bool = True,
        ax=None,
    ):
        focus_keys = self._normalize_focus_keys(focus)

        if not hasattr(data, label_attr):
            raise AttributeError(f"data has no attribute '{label_attr}'")

        runs = self._collect_gcn_runs(
            data,
            num_classes=num_classes,
            label_attr=label_attr,
            lrs=lrs,
            hidden_sizes=hidden_sizes,
            dropouts=dropouts,
            weight_decays=weight_decays,
            epochs=epochs,
            print_every=print_every,
        )

        return self._plot_gcn_runs(runs, focus=focus_keys, epochs=epochs, show=show, ax=ax)
