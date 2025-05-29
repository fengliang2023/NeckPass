# Artifacts for USENIX 2025 Paper
The files in this repository are for the USENIX 2025 paper:
NeckPass: Passive Authentication by Neck Cardiac Ballistocardiogram Biometrics in Virtual Reality/Augmented Reality Systems

We provide the following three compressed data packages:
1. **Experimental Results**: This package contains the data sources for all scenarios presented in the paper. It includes processed data and trained models. For each scenario, simply `python run.py` to obtain the data results shown in the paper.
1. **Source Code**: This package provides the source code for the technical approach mentioned in the paper. It includes data processing methods and model building code. By replacing the dataset with the corresponding scenario dataset mentioned in the paper and running `python run.py`, you can achieve the data processing results presented in the paper.
1. **Datasets**: This package includes the datasets for each scenario in the paper (in Mel-cepstral form, facilitating user privacy and a quick start), which can be used for training models and validating experimental results.

Detailed Explanation of the First Data Package: **Experimental Results**. The directory structure is as follows:

```
Result.xlsx
requirements.txt
8.2_OverallPerformance
├── Dataset_NeckPass
├── Model_Am
└── run.py
8.3.1_ImpactsofSingle-sideController
├── Dataset_NeckPass
└── run.py
8.3.2_ImpactsofIntensePhysicalActivity
├── Dataset_NeckPass
└── run.py
8.3.3_ImpactsofDynamicBehavior
├── Dataset_NeckPass
└── run.py
8.3.4_ImpactsofBiometricLong-termVariability
├── Dataset_NeckPass
└── run.py
8.3.5_ImpactsofVRDevices
├── Dataset_NeckPass
├── Model_Am
└── run.py
8.3.6_ImpactsofGenetics
├── Dataset_NeckPass
├── Model_Am
└── run.py
8.3.7_ImpactsofVotingMechanism
├── Dataset_NeckPass
└── run.py
8.3.8_EvaluationofComputationalDelay
├── Authentication
│ ├── Dataset_NeckPass
│ └── run.py
├── SignalRefinementAndConversion
│ ├── Dataset
│ ├── LibCode.py
│ ├── Preprocessing.py
│ ├── run.py
│ ├── SampleConversion.py
│ └── SignalRefinement.py
└── VRapp
└── neckpass.apk
9.2_ImperUser36ationAttack
├── Dataset_NeckPass
└── run.py
9.3_ReplayAttack
├── Dataset_NeckPass
└── run.py
9.4_SpoofingAttack
├── Dataset_NeckPass_Device
├── Dataset_NeckPass_Location
├── run_device.py
└── run_location.py
lib
├── ast_model.py
├── lib_siam.py
└── tpr.json
```

In the first-level directory, the `requirements.txt` file lists all the libraries required for this folder. To install the dependencies, execute the following command (using Python version 3.9.11):
`pip install -r requirements.txt `

The `Result.xlsx` file provides the data presented in each scenario of the paper, serving as a reference outline for readers. The `lib` folder contains implementations of library functions. The remaining folders correspond to the scenario resource files, named to align with the chapters in the paper for easy reference. These resource files include datasets (named `Dataset_NeckPass`), trained authentication models (`Model_Am`), and executable scripts (`run.py`). Users need only to navigate to the folder containing `run.py` and execute:
`python run.py`

This will yield the corresponding scenario data results, consistent with those provided in `Result.xlsx`.

**If you encounter any difficulties, please don't hesitate to reach out for assistance. Thank you sincerely for your interest and patience.**