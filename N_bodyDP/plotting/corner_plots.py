import matplotlib

matplotlib.use("Agg")
import corner


def corner_plot(result, truth=None, path=None, labels=None):
    names = list(result.names)
    labels = labels or names
    truths = None if truth is None else [truth[n] for n in names]
    fig = corner.corner(
        result.samples,
        weights=result.weights,
        labels=labels,
        truths=truths,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        bins=30,
    )
    if path:
        fig.savefig(path, dpi=200)
    return fig
