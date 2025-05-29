from Preprocessing import Preprocessing 
from SignalRefinement import SignalRefinement 
from SampleConversion import SampleConversion

# 读取文本文件
if __name__ == "__main__":
    directory = "Dataset"
    Preprocessing(directory)    # Preprocessing
    SignalRefinement(directory)    # Signal Refinement. The result will be saved in str(directory + "_filtered")
    SampleConversion(directory + "_filtered")   # Sample Conversion. The result will be saved in str(directory + "_filtered" + "_extract")
