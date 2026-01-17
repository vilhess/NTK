import seaborn as sns

def configure_seaborn(**kwargs):
    sns.set_context("notebook")
    sns.set_theme(sns.plotting_context("notebook", font_scale=1), style="whitegrid",
                  rc={
            # grid activation
            "axes.grid": True,
            # grid appearance
            "grid.color": "#BFBFBF",
            "axes.edgecolor": "#BFBFBF",
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.4,
            'font.family':'sans-serif',
            'font.sans-serif':['Lato'],
        },
        palette = ["#4F648F", "#E08042", "#54AB69", "#CE2F49", "#A26EBF", "#75523A", "#D12AA2", "#E0D06F", "#6F9AA7", "#3359C4",
               "#76455B"])

    