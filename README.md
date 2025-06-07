# clean_ppg_project
Clean version of PPG project - including code from start to finish. 


# How to: 
- Get pyPPG, and download it into the folder that it is supposed to go into (-> somewhere in preprocessing)
- Change paths in initialize.py, as they are hard coded and currently point to the data on my machine 
- Go into 02_clean_ppg, and run "python -m preprocessing.preprocessing" (run modules)


# Notes: 
- Six subjects did not have > 15 seconds recording >> were removed (see /Users/adrian/Documents/01_projects/02_clean_ppg/data/preprocessed/skipped_subjects.csv)
- Sampling frequency seems to have been 500 Hz... 


# Generally what happened: 
- Built a preprocessing pipeline, that works better than NeuroKit2 and PyPPG at extracting INDIVIDUAL wrist PWs. This pipeline works for both the MAUS and the AURORA data. 
- When using the MAUS dataset, one can validate that the custom algorithm works best for the wrist data (in comparison to the other two algorithms): This can be seen due to the distribution of the durations of the detected pulse waves; and the low Jensen-Shannon distance between the "gold-standard" duration distribution (assume: pyPPG and NeuroKit2 have for the FINGER data high quality in peak detection. This means that the distribution of the 


# To Dos: 
- Recreate Multivariate Regression. 
- Check out wave classification algorithm again.
- Figure out some nice DL things I can do with all the data I have? 
- Write really nice overviews about how to use it
- There could be more interesting stuff in diabetes patients... -> Include this again in next experiment? 
