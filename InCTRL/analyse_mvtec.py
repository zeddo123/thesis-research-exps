import seaborn as sns
import matplotlib.pyplot as plt

from easy_utils import AnomalyDataset

sns.set_theme(palette='pastel')


test_dataset = AnomalyDataset(
        dataset_name='mvtec',
        dataset_path='../datasets/mvtec',
        transform=None,
)
train_dataset = AnomalyDataset(
        dataset_name='mvtec',
        dataset_path='../datasets/mvtec',
        transform=None,
        split='train'
)

test_dataset.recap()
print()
train_dataset.recap()

df = test_dataset.get_df()

df.plot(kind='bar', title='Overview of Mvtec Test split')
plt.legend(loc="upper right", ncol=len(df.columns))

sns.displot(df)

plt.show()

