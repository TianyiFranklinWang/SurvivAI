import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TCGAMilGenomeSurvivalDataset(Dataset):
    def __init__(self, root, csv_path, splits, k=5, slide_id_column="slide_id", label_column="survival_months",
                 num_bins=4, shuffle_train_feature=True, genome_data_type=('mut', 'cnv', 'rnaseq')):
        self.root = root
        self.csv_path = csv_path
        self.splits = splits
        self.k = k
        self.fold_nb = 0
        self.slide_id_column = slide_id_column
        self.label_column = label_column
        self.num_bins = num_bins
        self.eps = 1e-6
        self.shuffle_train_feature = shuffle_train_feature
        self.genome_data_type = genome_data_type

        self.slide_data = pd.read_csv(self.csv_path, compression='zip', header=0, index_col=0, sep=',',
                                      low_memory=False)
        self.filter_data()
        self.num_classes = None
        self.gen_disc_label()
        self.slides = list(self.slide_data[self.slide_id_column])
        self.genome_data_column = [_ for _ in self.slide_data.columns if
                                   any(g_type in _ for g_type in self.genome_data_type)]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        self.train_patients = list()
        self.val_patients = list()
        self.train_slides_idx = list()
        self.val_slides_idx = list()
        self.used_slides_idx = list()
        self.update_fold_nb(0)

        self.shuffle = False

    def filter_data(self):
        slide_to_remove = list()
        for slide_id in self.slide_data[self.slide_id_column]:
            if not os.path.exists(os.path.join(self.root, f"{slide_id.split('.')[0]}.pt")):
                warnings.warn(f"Features for slide: {slide_id} does not exist")
                slide_to_remove.append(slide_id)
        if len(slide_to_remove) > 0:
            self.slide_data = self.slide_data[~self.slide_data[self.slide_id_column].isin(slide_to_remove)]
            self.slide_data = self.slide_data.reset_index(drop=True)

    def gen_disc_label(self):
        patients_df = self.slide_data.drop_duplicates(["case_id"]).copy()
        uncensored_df = self.slide_data[self.slide_data['censorship'] < 1]

        if len(uncensored_df) > 0:
            disc_labels, q_bins = pd.qcut(uncensored_df[self.label_column], q=self.num_bins, retbins=True, labels=False)
            q_bins[-1] = self.slide_data[self.label_column].max() + self.eps
            q_bins[0] = self.slide_data[self.label_column].min() - self.eps
            disc_labels, q_bins = pd.cut(patients_df[self.label_column], bins=q_bins, retbins=True, labels=False,
                                         right=False, include_lowest=True)
        else:
            disc_labels, q_bins = pd.cut(patients_df[self.label_column], bins=self.num_bins, retbins=True,
                                         labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        label_dict = dict()
        key_count = 0
        for i in range(len(q_bins) - 1):
            for c in [0, 1]:
                label_dict.update({(i, c): key_count})
                key_count += 1

        for i in range(len(self.slide_data)):
            key = patients_df.loc[patients_df['case_id'] == self.slide_data.iloc[i]['case_id']]['label'].item()
            self.slide_data.at[i, 'disc_label'] = key
            censorship_value = self.slide_data.at[i, 'censorship']
            key = (key, int(censorship_value))
            self.slide_data.at[i, 'label'] = label_dict[key]

        self.num_classes = len(label_dict)

    def update_fold_nb(self, fold_nb):
        self.fold_nb = fold_nb

        fold_csv = pd.read_csv(os.path.join(self.splits, f"splits_{fold_nb}.csv"), header=0, index_col=0, sep=',',
                               low_memory=False)
        self.train_patients = fold_csv['train']
        self.val_patients = fold_csv['val']

        self.train_slides_idx = self.slide_data[self.slide_data["case_id"].isin(self.train_patients)].index.tolist()
        self.val_slides_idx = self.slide_data[self.slide_data["case_id"].isin(self.val_patients)].index.tolist()

    def train(self):
        self.used_slides_idx = self.train_slides_idx
        self.shuffle = True if self.shuffle_train_feature else False

    def val(self):
        self.used_slides_idx = self.val_slides_idx
        self.shuffle = False

    def get_label(self, idx):
        return int(self.slide_data['label'][idx])

    def save_current_fold(self, save_path):
        fold_df = pd.DataFrame.from_dict({"train_case_id": self.train_patients, "test_case_id": self.val_patients},
                                         orient='index').transpose()
        fold_df.to_csv(save_path)

    def __len__(self):
        return len(self.used_slides_idx)

    def __getitem__(self, idx):
        slide = self.slides[self.used_slides_idx[idx]]

        feature = torch.load(os.path.join(self.root, f"{slide.split('.')[0]}.pt"))
        if self.shuffle:
            shuffled_indices = torch.randperm(feature.size(0))
            feature = feature[shuffled_indices]
        slide_series = self.slide_data.loc[self.slide_data[self.slide_id_column] == slide]
        assert len(slide_series) == 1, f"Multiple records exist for slide {slide}"
        genome_feature = torch.tensor(slide_series[self.genome_data_column].to_numpy().squeeze(), dtype=torch.float32)
        label = torch.tensor(slide_series['disc_label'].to_numpy(), dtype=torch.int)
        event_time = torch.tensor(slide_series[self.label_column].to_numpy())
        c = torch.tensor(slide_series['censorship'].to_numpy())

        return feature, genome_feature, label, event_time, c
