# The PPG Project - Describing PPG pulse wave shapes recorded at the wrist and their influences 


# How to: 
- (In theory: download pyPPG and put it into right folder. For now, the pyPPG folder is already embedded)
- Change paths in initialize.py, as they are hard coded and currently point to the data on my machine. Download raw data. 
- Run ```notebooks/00_preprocessing.ipynb``` to preprocess the AURORA-BP dataset
- Run ```notebooks/01_comparison_different_algorithms.ipynb``` to compare the preprocessing of pyPPG (developed for PPG signals recorded at the finger), NeuroKit2 (also developed for PPG signals recorded at the finger), and the preprocessing of the custom pipeline
- (For now, will be changed to the same style as the others: Run ```analysis/aurora_analysis.py``` to produce the multivariate analysis) 


# To-Dos: 
- The area under the curve is not introduced in the clean skripts here perfectly - the skript is currently using the "classes", to make the AUC positive or negative depending on whether the first or second peak is higher. This has it's issues, and will need to be re-introduced. 
- Clean up the output folder. 
- Make it a choice to produce either pdfs or pngs (or both). Currently, both are always created for ease of putting it into my slide-creation-application (pdf does not work here) and into latex (png is not vector based). 

# Want-to-Dos: 
- Seeing if (when age, bmi, etc are given) gender and fritzpartick scale make an significant difference in PW morphology. This is just out of pure interest.  
- There could be more interesting stuff in diabetes patients... -> Preprocess those subjects and put them in a seperate folder for analysis? 

# Notes: 
- Six subjects did not have > 15 seconds recording >> were removed (see /Users/adrian/Documents/01_projects/02_clean_ppg/data/preprocessed/skipped_subjects.csv)
- The code has been heavily cleaned up with the help of LLMs. 

