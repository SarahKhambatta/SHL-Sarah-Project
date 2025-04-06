# SHL-Sarah-Project
Build a Grammar Scoring Engine for Voice Samples (hosted on Kaggle).

This repository contains a Python-based Grammar Scoring Engine that processes voice samples to evaluate spoken English grammar. The engine performs speech-to-text conversion, detects grammar issues, and generates a grammar score based on the severity and frequency of errors.

## Features

- Converts voice samples (.wav or .mp3) to English text
- Checks grammar correctness using NLP tools
- Assigns a grammar score on a scale of 0 to 10
- Displays error suggestions for each transcription
- Outputs the results in a structured CSV report

## Technologies Used

- Python 3.x
- speech_recognition or Whisper (for speech-to-text conversion)
- language_tool_python (for grammar analysis)
- pandas (for data processing)
- os, json, and other built-in libraries

## How It Works

1. Upload voice samples into the `voice_samples` folder
2. Convert each audio file into a text transcript
3. Analyze the transcript for grammar issues
4. Generate a score based on the number of errors and word count
5. Store the results (filename, transcript, score, and suggestions) in a CSV file

## Project Structure

The objective of this competition is to develop a Grammar Scoring Engine for spoken data samples. You are provided with an audio dataset where each file is between 45 to 60 seconds long. The ground truth labels are MOS Likert Grammar Scores for each audio instance (see rubric below). Your task is to build a model that takes an audio file as input and outputs a continuous score ranging from 0 to 5.

##Project Architecture

grammar-scoring-engine/
├── notebooks/
│   └── Grammar_Scoring_Engine.ipynb
├── app/
│   ├── app.py
│   └── utils.py
├── models/
├── data/
│   ├── audios_train/
│   ├── audios_test/
│   ├── train.csv
│   └── test.csv
├── requirements.txt
└── README.md
