import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from easy_utils import AnomalyDataset

sns.set_theme(palette='pastel')


test_dataset = AnomalyDataset(
        dataset_name='visa',
        dataset_path='visa_anomaly_detection/visa',
        transform=None,
)
train_dataset = AnomalyDataset(
        dataset_name='visa',
        dataset_path='visa_anomaly_detection/visa',
        transform=None,
        split='train'
)

test_dataset.recap()
print()
train_dataset.recap()

df = test_dataset.get_df()

df.plot(kind='bar', title='VisA Dataset Test split')
plt.legend(loc="upper right", ncol=len(df.columns))

plt.xticks(rotation=45, ha='right')

df_train = train_dataset.get_df()
print(df_train['anomaly-free'])
print(sum(df_train['anomaly-free']))

plt.show()

