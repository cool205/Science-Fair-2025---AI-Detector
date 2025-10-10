import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath='best_per_run_summary.csv'):
    df = pd.read_csv(filepath)
    df = df.drop(columns=['epoch'], errors='ignore')
    return df

def faceted_heatmaps(df):
    df['lr_bin'] = pd.cut(df['learning_rate'], bins=3)
    df['cj_bin'] = pd.cut(df['color_jitter'], bins=3)
    g = sns.FacetGrid(df, col='lr_bin', row='cj_bin', margin_titles=True)
    g.map_dataframe(
        lambda data, color: sns.heatmap(
            data.pivot_table(index='dropout', columns='batch_size', values='val_acc'),
            cmap='viridis', annot=True, fmt=".2f", cbar=False
        )
    )
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Faceted Heatmaps of val_acc by Dropout and Batch Size\nGrouped by Learning Rate and Color Jitter')
    plt.show()

def grid_heatmap(df):
    # Use actual values as categories for cleaner labels
    df['lr_cat'] = df['learning_rate'].astype(str)
    df['cj_cat'] = df['color_jitter'].astype(str)

    lr_vals = sorted(df['lr_cat'].unique())
    cj_vals = sorted(df['cj_cat'].unique())

    fig, axes = plt.subplots(len(cj_vals), len(lr_vals), figsize=(2.5 * len(lr_vals), 2.2 * len(cj_vals)), squeeze=False)

    for i, cj in enumerate(cj_vals):
        for j, lr in enumerate(lr_vals):
            ax = axes[i][j]
            subset = df[(df['lr_cat'] == lr) & (df['cj_cat'] == cj)]
            pivot = subset.pivot_table(index='dropout', columns='batch_size', values='val_acc')

            sns.heatmap(pivot, ax=ax, cmap='RdBu_r', annot=True, fmt=".3f", cbar=False,
                        vmin=df['val_acc'].min(), vmax=df['val_acc'].max(),
                        annot_kws={"size": 6}, linewidths=0.5, linecolor='gray')

            ax.tick_params(axis='both', labelsize=8)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=8)

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(0.8)

            # Add column labels (learning rate) only once at top row
            if i == 0:
                ax.set_title(f"Learning Rate: {lr}", fontsize=8)

            # Add row labels (color jitter) only once at leftmost column
            if j == 0:
                ax.annotate(f"Color Jitter: {cj}", xy=(0, 0.5), xytext=(-0.5, 0.5),
                            textcoords='axes fraction', fontsize=8, rotation=90,
                            va='center', ha='center')

            # Add axis labels inside every subplot
            ax.set_xlabel("Batch Size", fontsize=6)
            ax.set_ylabel("Dropout", fontsize=6)

    plt.tight_layout(pad=1.0)
    plt.suptitle("Validation Accuracy by Dropout & Batch Size\nGrouped by Learning Rate and Color Jitter", fontsize=10, y=1.05)
    plt.show()
    

def parallel_plot(df):
    cols = ['dropout', 'batch_size', 'learning_rate', 'color_jitter', 'val_acc']
    df_norm = df[cols].copy()
    scaler = MinMaxScaler()
    df_norm[cols] = scaler.fit_transform(df_norm[cols])
    df_norm['val_acc_group'] = pd.cut(df['val_acc'], bins=3, labels=['Low', 'Medium', 'High'])
    plt.figure(figsize=(14,6))
    parallel_coordinates(df_norm, 'val_acc_group', colormap=plt.get_cmap("Set1"))
    plt.title('Parallel Coordinates: Hyperparameters and Validation Accuracy')
    plt.grid(True)
    plt.show()

def scatter_3d(df):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df['dropout'], df['learning_rate'], df['val_acc'],
                    c=df['batch_size'], s=df['color_jitter']*100, cmap='viridis')
    ax.set_xlabel('Dropout')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel('Validation Accuracy')
    plt.title('3D Scatter: Dropout, LR, val_acc (Color: Batch Size, Size: Color Jitter)')
    plt.colorbar(sc, label='Batch Size')
    plt.show()

def pairplot_all(df):
    cols = ['dropout', 'batch_size', 'learning_rate', 'color_jitter', 'val_acc']
    sns.pairplot(df[cols], corner=True, diag_kind='kde', palette='viridis')
    plt.suptitle('Pairplot of Hyperparameters and Validation Accuracy', y=1.02)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hyperparameter interactions with validation accuracy")
    parser.add_argument('--facetheatmap', action='store_true', help='Show faceted heatmaps')
    parser.add_argument('--gridheatmap', action='store_true', help='Show rectangular grid of heatmaps')
    parser.add_argument('--parallel', action='store_true', help='Show parallel coordinates plot')
    parser.add_argument('--scatter3d', action='store_true', help='Show 3D scatter plot')
    parser.add_argument('--pairplot', action='store_true', help='Show pairplot')

    args = parser.parse_args()
    df = load_data()

    if args.facetheatmap:
        faceted_heatmaps(df)
    if args.gridheatmap:
        grid_heatmap(df)
    if args.parallel:
        parallel_plot(df)
    if args.scatter3d:
        scatter_3d(df)
    if args.pairplot:
        pairplot_all(df)
