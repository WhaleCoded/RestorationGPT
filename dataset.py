import torch
from torch.utils.data import Dataset
import os
import json
from typing import List, Dict, Any


class RestorationData(Dataset):
    def __init__(self, root: str, length: int = None):
        self.root = root
        self.length = length
        prepared_data = self.prepare_data(root)
        if length is not None:
            prepared_data = prepared_data[:length]
        self.data = prepared_data

    def _convert_clean_json_entry(self, entry: dict) -> dict:
        cleaned_data = ""
        for key, value in entry.items():
            cleaned_data += key + ":\n"
            cleaned_data += value + "\n"

        return cleaned_data

    def prepare_data(self, root: str) -> List[Dict[str, Any]]:
        data_folders = os.listdir(root)
        json_files = []
        for folder in data_folders:
            folder_path = os.path.join(root, folder)
            if os.path.isfile(folder_path) and folder_path.endswith(".json"):
                json_files.append(folder_path)
            elif os.path.isdir(folder_path):
                possible_json_files = os.listdir(folder_path)
                for file in possible_json_files:
                    file_path = os.path.join(folder_path, file)
                    if file.endswith(".json") and os.path.isfile(file_path):
                        json_files.append(os.path.join(file_path))
        print(json_files)

        prepared_data = []
        for file in json_files:
            with open(file, "r") as f:
                data = json.load(f)
                for entry in data.values():
                    cleaned_entry = self._convert_clean_json_entry(entry)
                    prepared_data.append(cleaned_entry)

        return prepared_data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> dict:
        return self.data[index]


if __name__ == "__main__":
    ROOT_PATH = os.path.join(os.path.curdir, "data")

    dataset = RestorationData(ROOT_PATH)

    print(dataset[0])
    print(dataset[1])

    print(len(dataset))
