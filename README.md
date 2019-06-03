# Instagram-likes-predictions
This repository documents our team's journey in IEOR 135 Data-X

# Project Aim
- This project aims to create a ML model consisting of NNs and NLP to analyze Instagram Posts and suggest the best time and day to upload the photo as well as possible hashtags or photo filters to add.
- This is targeted towards social media influencers & marketing agencies to maximise positive impressions per post.

# How to run
1. Clone the repository to computer
2. In root folder, run command ```python LinReg.py```. This will take inputs from ```/dataxuserinput.csv``` and give a prediction.

# Our Journey
Previous models that we have tried can be found in ```/models```. We first webscraped with codes from [Instagram-scraper](https://github.com/rarcega/instagram-scraper). We subsequently attemped a naive ML model before deciding on using YOLO-object detection. After trying our models such as Gradient Boosting Ensemble, Random Forest, KNN, Voting etc. , we found that our Linear Regression model works best. 

# YOLO Object Detection
Below is an illustration of how YOLO object detection works to score an image\
<img src="/photo1.jpeg"  width="200" >
<img src="/photo2.jpeg"  width="200" >
<img src="/photo3.jpeg"  width="200" >
<img src="/photo4.jpeg"  width="200" >

