import torch
from torch.utils.data import WeightedRandomSampler


def visual_omics_collate_fn(batch):
    visual_features = [item[0] for item in batch]
    omics_features = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    event_times = [item[3] for item in batch]
    censorships = [item[4] for item in batch]

    return visual_features, omics_features, labels, event_times, censorships


def class_balanced_sampler(dataset):
    def make_weights_for_balanced_classes(dataset):
        class_counts = [len(cls_ids) for cls_ids in dataset.slide_cls_ids]
        total_samples = len(dataset)
        weight_per_class = [total_samples / count for count in class_counts]
        weights = [weight_per_class[dataset.get_label(idx)] for idx in range(total_samples)]
        return torch.DoubleTensor(weights)

    weights = make_weights_for_balanced_classes(dataset)
    return WeightedRandomSampler(weights, len(weights))
