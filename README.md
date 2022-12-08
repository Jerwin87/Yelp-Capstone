# Data Science Bootcamp Capstone Project 'ReView. Rating your Reviews'

## It is still a work-in-progress as the notebooks still have to be commented and corrected

This repository represents the capstone project of Jan, Felix, Max and me (Jérémy).

This project is was developed as final exercise to apply all the different techniques we learned during our data science bootcamp.

The goal of the project was to use state-of-the-art machine learning, to automate the ranking of old and new user reviews. 

For data we used the reviews from Yelp that are freely accessible through the Yelp-dataset under the following url: [Yelp-Dataset](https://www.yelp.com/dataset).

## the structure of this repo

This repo is structured by several jupyter notebooks which represent different steps of the data science life circle

* In a first step we had to import and preproces the data

* Then we did EDA

* Then we modelled different models via scikit-learn

* Then we build a Bi-LSTM Deep Recurring Neural Network

* Finally we created a very simple offline dashboard to present our results


* On top of that we feature engineered several aspects of language data, which all were not included in the final model but are documented in a final neotebook highlighting several NLP methods. 

## Requirements:

- pyenv with Python: 3.9.8

### Setup

Use the requirements file in this repo to create a new environment.

```
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
