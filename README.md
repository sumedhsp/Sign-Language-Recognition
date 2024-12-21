## Prerequisites
1. **Python Environment**:
   - Ensure you have Python 3.x installed.
   - Install the required dependencies using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

2. **Download I3D Weights**:
   - Download the appropriate I3D model weights from [this link](https://drive.google.com/file/d/1jALimVOB69ifYkeT0Pe297S1z4U3jC48/view?usp=sharing).
   - Place the downloaded weights file (.pt) in the `CV_Project_Files/` directory.

3. **Download Dataset**:
   - Download the dataset from [this link](https://drive.google.com/file/d/1UKESPYEvFsrrQByrl9mWny_JJdoMXKfk/view?usp=drive_link). This dataset was obtained directly from the author of the [original paper](https://github.com/dxli94/WLASL) so please use this for educational purposes only. To download the data for running the project on your own, please follow the original author's github repo for more instructions.
   - Create a folder named `data` inside the `CV_Project_Files/` directory and paste the dataset into it:
     ```
     CV_Project_Files/
     ├── data/
     │   ├── [dataset files here]


## Running the Project
### Training
- To train the model, use one of the training scripts. For example:
  `python train_vanilla_v1_2000.py`

### Testing
- To evaluate the model, use one of the testing scripts. For example:
  `python test_vanilla_v1_2000.py`


## Notes
- Ensure the preprocess/ and configfiles/ folders contain necessary configuration files for the model and preprocessing pipeline.

- If you encounter any issues, verify that all dependencies are installed and that the dataset and weights are correctly placed.