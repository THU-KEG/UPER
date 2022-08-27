import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

fre_shot_led = pd.DataFrame({
    'Training data proportion (%)': ['0',
                                     '0.01',
                                     '0.1',
                                     '1',
                                     '10',
                                     ] * 3,
    'Rouge': [
        15.73633333, 25.80566667, 30.59633333, 33.041, 35.57233333,
        14.005,
        25.05566667,
        29.57,
        32.73166667,
        35.48266667,
        15.79533333,
        23.40066667,
        27.38166667,
        30.522,
        33.99933333,
    ],
    'extractor': [
        # 'POER', 'POER', 'POER', 'POER', 'POER',
        'UPER', 'UPER', 'UPER', 'UPER', 'UPER',
        'tf-idf', 'tf-idf', 'tf-idf', 'tf-idf', 'tf-idf',
        'random', 'random', 'random', 'random', 'random',
    ]
})
human_evaluation = pd.DataFrame({
    'Domain': ['Animal',
               'Company',
               'Film',
               ] * 2,
    'Human preference (%)': [56.7,
                             58.3,
                             50.8,
                             43.3,
                             41.7,
                             49.2,
                             ],
    'extractor': [
        'UPER', 'UPER', 'UPER',
        'tf-idf', 'tf-idf', 'tf-idf',
    ]
})
# x_name = "Domain"
x_name = "Training data proportion (%)"
# y_name = "Human preference (%)"
y_name = "Rouge"
# pic_name = "human_evaluation"
pic_name = "fre_shot_led"
save_path = f"/home/tsq/TopCLS/src/statistics/pictures/histogram/{pic_name}.png"
f, ax = plt.subplots(figsize=(869 / 85, 513 / 85))
sns.barplot(x=x_name, y=y_name, hue="extractor", data=fre_shot_led)
plt.xticks(fontsize=20)  # x轴刻度的字体大小（文本包含在pd_data中了）
plt.yticks(fontsize=20)  # y轴刻度的字体大小（文本包含在pd_data中了）
plt.xlabel(x_name, fontdict={'weight': 'normal', 'size': 22})
plt.ylabel(y_name, fontdict={'weight': 'normal', 'size': 22})
plt.legend(title="extractor", fontsize=14, title_fontsize=14)
plt.show()
plt.close()
f.savefig(save_path, dpi=100, bbox_inches='tight')
