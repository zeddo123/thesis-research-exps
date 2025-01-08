import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from easy_utils import AnomalyDataset

sns.set_theme(palette='pastel')


test_dataset = AnomalyDataset(
        dataset_name='chexpert',
        dataset_path='../datasets/chexpert_mvtec',
        transform=None,
)
train_dataset = AnomalyDataset(
        dataset_name='chexpert',
        dataset_path='../datasets/chexpert_mvtec',
        transform=None,
        split='train'
)

df = test_dataset.get_df()
df_train = train_dataset.get_df()

#plt.legend(loc="upper right", ncol=len(df.columns))

#plt.xticks(rotation=45, ha='right')


data = {
    'class': ['anomaly-free', 'anomalous', 'anomaly-free', 'anomalous'],
    'examples': [sum(df_train['anomaly-free']), sum(df_train['anomalous']), sum(df['anomaly-free']), sum(df['anomalous'])],
    'split': ['train', 'train', 'test', 'test']
}

full_df = pd.DataFrame(data)
print(full_df)

ax = sns.barplot(x='split', y='examples',
            data=full_df, palette='pastel', legend='full', hue='class')
ax.set(title='Chexpert Dataset Breakdown by class')
for container in ax.containers:
    ax.bar_label(container)

"""
for i, v in enumerate(data['examples']):
   ax.text(i, v + 0.2, f'{v}', ha='center', color='black', fontsize=13)
"""

plt.show()
