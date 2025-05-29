from Preprocessing import Preprocessing 
from SignalRefinement import SignalRefinement 
from SampleConversion import SampleConversion
import time

def main():
    directory = "Dataset"
    Preprocessing(directory)    
    SignalRefinement(directory)    
    SampleConversion(directory + "_filtered")   

if __name__ == "__main__":
    start_time = time.time()

    main()

    end_time = time.time()
    duration = end_time - start_time
    print(f"Total execution time(120s): {duration:.2f} seconds")
    print(f"One-second sample execution time: {(duration/120):.2f} seconds")
