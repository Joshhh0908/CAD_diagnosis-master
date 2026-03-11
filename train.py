from dataprep import CPRDataset
from framework import sc_net_framework




def main():
    
    framework = sc_net_framework(pattern="pre_training", state_dict_root=None)
    model = framework.model
    dataLoader_train = framework.dataLoader_train

    for epoch in range(10):
        for batch in dataLoader_train:
            print(batch["volume"].shape, batch["stenosis"].shape, batch["plaque"].shape, batch["valid"].shape)
            break