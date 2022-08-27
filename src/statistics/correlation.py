import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import logging

ext_abs_rouge_wikicatsum = [
    {"ext": [0.65178, 0.24781, 0.42095], "abs": [0.39612, 0.19581, 0.31437], "source": "cheating_r1_recall_fs500"},
    {"ext": [0.64403, 0.288, 0.41905], "abs": [0.41142, 0.22119, 0.33195], "source": "cheating_r2_recall_fs500"},
    {"ext": [0.64763, 0.25598, 0.42227], "abs": [0.40161, 0.20533, 0.32344], "source": "cheating_rl_recall_fs500"},
    {"ext": [0.59411, 0.20006, 0.39574], "abs": [0.36529, 0.17279, 0.29327], "source": "f1_regression_lgb_fs500"},
    {"ext": [0.53417, 0.1676, 0.43774], "abs": [0.37359, 0.18554, 0.30696], "source": "tf_idf_fs500"},
    {"ext": [0.54611, 0.17816, 0.36147], "abs": [0.38157, 0.19721, 0.31562], "source": "inverse_nz_concat_fs500"},
    # {"ext": [0.65947, 0.27054, 0.43055], "abs": [0.4062, 0.20686, 0.32237], "source": "cheating_recall_avg_fs500"},
    # {"ext": [0.66499, 0.27667, 0.43256], "abs": [0.41768, 0.21899, 0.33365], "source": "cheating_f1_avg_fs500"},
    # {"ext": [0.49875, 0.13376, 0.40796], "abs": [0.34345, 0.15449, 0.27571], "source": "none_prompt_fs500"},
    # {"ext": [0.55928, 0.17401, 0.4587], "abs": [0.35608, 0.17148, 0.29109], "source": "inverse_prompt_fs500"},
    # {"ext": [0.55979, 0.17737, 0.45946], "abs": [0.35612, 0.16985, 0.29119], "source": "qa_prompt_fs500"},
    # {"ext": [0.42744, 0.14687, 0.28512], "abs": [0.36849, 0.18245, 0.3011], "source": "inverse_nz_fs500"},
]

ext_abs_rouge_wcep = [
    {"ext": [], "abs": [], "source": ""},
]


def work(args, ext_abs_rouges):
    data = []
    for ext_abs_rouge in ext_abs_rouges:
        data.append(ext_abs_rouge["ext"] + ext_abs_rouge["abs"])
    df = pd.DataFrame(data, columns=['R1-ext', 'R2-ext', 'RL-ext', 'R1-abs', 'R2-abs', 'RL-abs'])
    print(df)
    save_dir = "/home/tsq/TopCLS/src/statistics/pictures/corr"
    pic_name = f"corr_{args.type}_{args.data}_num{len(data)}_{args.color}"
    draw_corr(save_dir, pic_name, df, args.color)


def draw_corr(save_dir, pic_name, data_df, color='YlGnBu'):
    # heatmap
    save_path = os.path.join(save_dir, pic_name)
    corr = data_df.corr()
    print(corr)
    f, ax = plt.subplots(figsize=(20, 10))
    ax.xaxis.set_label_position('top')
    sns.heatmap(corr, cmap=color, linewidths=0.1, ax=ax, annot=True)
    plt.xticks(fontsize=25)  # x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=25, rotation=45)  # y轴刻度的字体大小（文本包含在pd_data中了）
    plt.show()
    plt.close()
    f.savefig(save_path, dpi=100, bbox_inches='tight')


if __name__ == "__main__":
    datasets = {
        'wikicatsum': ext_abs_rouge_wikicatsum,
        'wcep': ext_abs_rouge_wcep,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='wikicatsum', choices=datasets.keys())
    parser.add_argument('--type', type=str,
                        default='pearson'
                        )
    parser.add_argument('--color', type=str,
                        default='YlGnBu',
                        choices=['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu',
                                 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens',
                                 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn',
                                 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG',
                                 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r',
                                 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu',
                                 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r',
                                 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
                                 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r',
                                 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr',
                                 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper',
                                 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare',
                                 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat',
                                 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r',
                                 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2',
                                 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire',
                                 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako',
                                 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r',
                                 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r',
                                 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r',
                                 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r',
                                 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r',
                                 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
                        )
    args = parser.parse_args()
    work(args, datasets[args.data])
