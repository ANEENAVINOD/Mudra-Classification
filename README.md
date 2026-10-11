# Bharatanatyam Mudra Recognition

A CNN that recognizes 5 Bharatanatyam hand mudras from an image. A Streamlit app shows the predicted mudra, the confidence and the probability of every class.

## Files
- mudra_classification.ipynb : full notebook (analysis, training, evaluation)
- app.py : Streamlit app
- models/ : trained model (.keras), label encoder and model info
- screenshots/ : plots and app screenshots
- requirements.txt : libraries needed

## How to run
1. Install libraries:
   pip install -r requirements.txt
2. Open the notebook and run all cells (needs mudra.zip, not included here).
3. Start the app:
   streamlit run app.py
