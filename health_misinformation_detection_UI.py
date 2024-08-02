import streamlit as st
import pickle
import numpy as np
from config import corona_best_model, maternity_best_model, lassafever_best_model


class Classify_Health_Info:
    def __init__(self, model_name, text=None):
        self.model_name = model_name
        self.text = text

    def load_model(self):
        '''
        Function to load model, given the name of the model, and
        return the model

        args:
        name: name of model could be 'Corona_Model', 'Lassafever_Model' or 'Maternity_Health_Model'

        returns: Loaded model in pkl format
        '''
        save_model_dir = "path_to_saved_models/"  # Update with the correct path

        if self.model_name == 'Corona_Model':
            with open(corona_best_model, 'rb') as f:
                corona_model = pickle.load(f)
            return corona_model

        elif self.model_name == 'Lassafever_Model':
            with open(lassafever_best_model, 'rb') as f:
                lassa_model = pickle.load(f)
            return lassa_model

        elif self.model_name == 'Maternity_Health_Model':
            with open(maternity_best_model, 'rb') as f:
                materniy_model = pickle.load(f)
            return materniy_model
        else:
            st.error(
                'Unknown Model Name Parameter, Model name could either be Corona_Model, Lassafever_Model or Maternity_Health_Model')
            return None

    def predict(self):
        model = self.load_model()
        if model:
            preds = model.predict([self.text])
            pred_proba = model.predict_proba([self.text])
            return preds, pred_proba
        else:
            return None, None

# Streamlit app


def main():
    st.title("Health Misinformation Detection")

    # User input for the case type
    case_type = st.selectbox(
        "Select the type of case:",
        ("Coronavirus", "Lassa fever", "Maternity health")
    )

    # Mapping user-friendly case type to model name
    case_type_to_model_name = {
        "Coronavirus": "Corona_Model",
        "Lassa fever": "Lassafever_Model",
        "Maternity health": "Maternity_Health_Model"
    }

    # User input for text
    user_input = st.text_area("Enter the text to classify:")

    # Predict button
    if st.button("Predict"):
        if user_input:
            model_name = case_type_to_model_name[case_type]
            classifier = Classify_Health_Info(model_name, user_input)
            prediction, prediction_proba = classifier.predict()
            if prediction is not None:
                pred_class = 'True' if prediction else 'False'
                pred_proba = prediction_proba[0][1] if prediction else prediction_proba[0][0]
                pred_proba_percent = round(pred_proba * 100, 2)

                if prediction:
                    st.success(
                        f"The prediction is: {pred_class} with a confidence of {pred_proba_percent}%")
                else:
                    st.error(
                        f"The prediction is: {pred_class} with a confidence of {pred_proba_percent}%")

                # Add additional interactive elements
                if pred_proba_percent >= 75:
                    st.success("High confidence in the prediction.")
                    st.progress(pred_proba_percent/100)
                else:
                    st.info(
                        "Confidence is moderate. Consider verifying with additional sources.")
            else:
                st.write("Model loading failed.")
        else:
            st.error("Please enter text to classify.")


if __name__ == "__main__":
    main()
