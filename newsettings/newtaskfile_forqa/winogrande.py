import os
import datasets
import numpy as np

from fewshot_gym_dataset import FewshotGymDataset, FewshotGymTextToTextDataset

class Winogrande(FewshotGymTextToTextDataset):
    def __init__(self):
        self.hf_identifier = "wino_grande"
        self.task_type = "text-to-text"


    def map_hf_dataset_to_list(self, hf_dataset, split_name):
        lines = []
        for datapoint in hf_dataset[split_name]:
            # line[0]: input; line[1]: output
            if int(datapoint["answer"]) == 1:
                lines.append((datapoint["sentence"] + " (A) " + datapoint["option1"] + " (B) " + datapoint["option2"], datapoint["option1"]))
            elif int(datapoint["answer"]) == 2:
                lines.append((datapoint["sentence"] + " (A) " + datapoint["option1"] + " (B) " + datapoint["option2"], datapoint["option2"]))
        return lines

    def get_train_test_lines(self, dataset):
        train_lines = self.map_hf_dataset_to_list(dataset, "train")
        test_lines = self.map_hf_dataset_to_list(dataset, "validation")
        if len(test_lines) > 1000:
            test_lines = test_lines[0:1000]
        return train_lines, test_lines

    def load_dataset(self):
        return datasets.load_dataset('winogrande', 'winogrande_xl')

def main():
    dataset = Winogrande()

    for seed in [100, 13, 21, 42, 87]:
        train, dev, test = dataset.generate_k_shot_data(k=32, seed=seed, path="../data/")

if __name__ == "__main__":
    main()