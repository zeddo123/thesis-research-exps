import seaborn as sns
import matplotlib.pyplot as plt

from easy_utils import AnomalyDataset

sns.set_theme(palette='pastel')


test_dataset = AnomalyDataset(
        dataset_name='riseholme',
        dataset_path='../datasets/riseholme_mvtec',
        transform=None,
)
train_dataset = AnomalyDataset(
        dataset_name='riseholme',
        dataset_path='../datasets/riseholme_mvtec',
        transform=None,
        split='train'
)

test_dataset.recap()
print()
train_dataset.recap()

df = test_dataset.get_df()

df.plot(kind='bar', title='Overview of Riseholme Test split')
plt.legend(loc="upper right", ncol=len(df.columns), fontsize=12)

plt.xticks(rotation=0, ha='right', fontsize=13)

df_train = train_dataset.get_df()
print(df_train['anomaly-free'])
print(sum(df_train['anomaly-free']))

plt.show()
