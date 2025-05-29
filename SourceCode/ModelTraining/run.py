from DatasetSplit import DatasetSplit
from TrainTm import TrainTm
from ExtractRepresentation import ExtractRepresentation
from TrainAm import TrainAm
from NeckPass import NeckPass

# 读取文本文件
if __name__ == "__main__":
    InputDataset = "Dataset"
    SplitDataset = "DatasetSplit"
    SaveModelPath = "SaveModel"
    RepresentationPath = "Representation"
    DatasetSplit(InputDataset, SplitDataset)    #  Divide the training set and test set
    TrainTm(SplitDataset, SaveModelPath)    #  Training Tm
    ExtractRepresentation(SplitDataset, RepresentationPath, SaveModelPath)    #  Representation Extraction for training Am
    TrainAm(RepresentationPath)    #  Training Am
    NeckPass(SaveModelPath)    #  Calculating TPR, TNR, BAC
