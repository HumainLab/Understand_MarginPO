import os
import json
import numpy as np
from lets_plot import *

LetsPlot.setup_html()

def plot_heatmap(sample_idx, cosine_sim_matrix, pos_tokens, neg_tokens, metadata, output_dir='plots'):
    import pandas as pd

    # Prepare data for plotting
    df = pd.DataFrame(cosine_sim_matrix, columns=neg_tokens, index=pos_tokens)
    df = df.reset_index().melt(id_vars='index')
    df.columns = ['Chosen Token', 'Rejected Token', 'Cosine Similarity']

    # Create the heatmap
    plot = (
        ggplot(df, aes('Rejected Token', 'Chosen Token', fill='Cosine Similarity'))
        + geom_tile(aes(color='Cosine Similarity'))
        + scale_fill_gradient2(low='blue', mid='white', high='red', midpoint=0)
        + scale_color_gradient2(low='blue', mid='white', high='red', midpoint=0)
        + ggsize(800, 600)
        # + ggtitle(
        #     f"Training Step {metadata['step']}: "
        #     f"Token-Level Correlation Heatmap"
        #     # f"Cosine Similarity (Response): {metadata['cosine_sim_response']:.2f}, "
        #     # f"Correlation Min: {metadata['correlation_min']:.2f}"
        #     )
        + labs(x='Rejected', y='Chosen')
        + theme(
        axis_text_x=element_text(angle=30, hjust=1),
        axis_text=element_text(size=12, face='bold')
    )
    )

    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ggsave(plot, os.path.join(output_dir, f'heatmap_{sample_idx}.pdf'))
    print(f"Saved heatmap for sample {sample_idx} to lets-plot-images/{output_dir}/heatmap_{sample_idx}.pdf")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default='results', help="Input directory containing result files")
    parser.add_argument("--output_dir", type=str, default='plots', help="Output directory for plots")
    args = parser.parse_args()

    input_files = os.listdir(args.input_dir)
    sample_indices = set()
    for file_name in input_files:
        if file_name.startswith('sample_') and file_name.endswith('_metadata.json'):
            idx = int(file_name.split('_')[1])
            sample_indices.add(idx)

    for idx in sorted(sample_indices):
        # Load metadata
        metadata_path = os.path.join(args.input_dir, f'sample_{idx}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Load cosine similarity matrix
        cosine_sim_matrix_path = os.path.join(args.input_dir, f'sample_{idx}_cosine_sim_matrix.npy')
        cosine_sim_matrix = np.load(cosine_sim_matrix_path)

        # Get tokens
        pos_tokens = metadata['pos_tokens']
        neg_tokens = metadata['neg_tokens']
        metadata["step"] = args.input_dir.split('/')[-1]

        # Plot heatmap
        plot_heatmap(idx, cosine_sim_matrix, pos_tokens, neg_tokens, metadata, output_dir=args.output_dir)

if __name__ == '__main__':
    main()