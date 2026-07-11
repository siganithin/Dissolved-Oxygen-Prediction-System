from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.set_text_color(0, 51, 153)
        self.cell(0, 10, 'Expert Viva Preparation Guide: DO Prediction System', align='C')
        self.ln(15)

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(0, 51, 153)
        self.cell(0, 10, title, ln=True)
        self.ln(2)

    def question(self, q_text):
        self.set_font('helvetica', 'B', 12)
        self.set_text_color(192, 0, 0)
        self.multi_cell(0, 8, f"Q: {q_text}")
        
    def answer(self, a_text):
        self.set_font('helvetica', '', 11)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 6, f"A: {a_text}")
        self.ln(6)
        
    def bullet_bold(self, title, text):
        self.set_font('helvetica', 'B', 11)
        self.set_text_color(0, 0, 0)
        self.write(6, f"• {title}: ")
        self.set_font('helvetica', '', 11)
        self.write(6, f"{text}\n")
        self.ln(2)

def main():
    pdf = PDF()
    pdf.add_page()
    
    # Intro
    pdf.set_font('helvetica', 'I', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, "This document is an expert-level preparation guide designed specifically for your PPT explanation and Viva Voice. It breaks down all the technical concepts from UML Diagrams to Model accuracies in a humanized way.")
    pdf.ln(10)
    
    # Section 1: Dashboard Flow & Mechanics
    pdf.chapter_title("1. Deep Dive: Dashboard & Streamlit Explanation")
    pdf.question("Why did you choose Streamlit over other web frameworks like React or Django?")
    pdf.answer("Because our core architecture is deeply tied to Python Data Science libraries (NumPy, Pandas, TensorFlow). Streamlit allows us to build native Machine Learning frontends directly from Python. It has built-in handlers for Pandas DataFrames and Matplotlib charts, making it incredibly fast to deploy without needing complex REST APIs or frontend HTML/CSS code.")

    pdf.question("How did you optimize the dashboard so it doesn't freeze or slow down?")
    pdf.answer("Streamlit runs the entire Python script from top to bottom every time a user drags a slider. If we didn't optimize it, our heavy BiSRU Deep Learning model would reload and predict thousands of rows every time the slider was touched! To fix this, I strictly used the @st.cache_data decorator. Now, when a CSV is uploaded, predictions are run exactly once and cached in local memory. Slider adjustments are completely instant because they only filter the cached data.")

    # Section 2: UML Diagrams
    pdf.chapter_title("2. Explaining Project UML Diagrams (For your PPT)")
    pdf.question("How do you explain the Use Case Diagram?")
    pdf.answer("The primary actor is the 'Aquaculturist' (farmer). Their visible use cases are 'Upload Data', 'View Dashboard', and 'Receive Alerts'. However, our back-end actions like 'Preprocess Data' and 'Feature Selection' are modeled with an <<includes>> relationship because they execute automatically as mandatory sub-steps whenever the farmer uploads data.")

    pdf.question("What does the Activity Diagram represent?")
    pdf.answer("It shows the step-by-step real-world flow. Flow starts at User Upload -> Cleans Data -> Maps to LightGBM to pick top 4 features -> BiSRU predicting DO level over 24-time steps -> Finally hitting a Decision Diamond ('Is DO < Threshold?'). True branches to generating visual warnings, False loops back to normal status.")
    
    pdf.question("How do you summarize the System Architecture?")
    pdf.answer("It's a pure modular sequential pipeline structure. Layer 1: Data Input. Layer 2: Preprocessing and Normalization. Layer 3: Feature Extraction (LightGBM). Layer 4: Deep Learning Inference (BiSRU+Attention). Layer 5: Presentation/Dashboard. The modular design ensures that if we update the Neural Network, the UI doesn't break at all.")

    # Section 3: Machine Learning & Preprocessing
    pdf.chapter_title("3. Machine Learning & Preprocessing Basics")
    
    pdf.question("Why didn't you just use simple ML algorithms like Random Forest?")
    pdf.answer("Random Forest acts on tabular rows as if they are independent. But Dissolved Oxygen is a 'Time Series'. The oxygen level at 3 PM heavily depends on the level at 2 PM. Deep Learning Recurrent Networks (like BiSRU) contain 'sequential memory' and map past chronological dependencies, which traditional ML models simply cannot do.")

    pdf.question("What exactly is Min-Max Normalization and why is it mandatory?")
    pdf.answer("In real life, Temperature is in 30s, but pH is 7, and Ammonia is 0.02. A neural network naturally favors bigger numbers and would completely ignore Ammonia despite it being highly toxic to fish. Min-Max Normalization mathematically crushes every feature into a 0.0 to 1.0 range, ensuring the neural network evaluates all parameters with equal, unbiased importance.")

    pdf.question("What role does LightGBM play since you already have BiSRU?")
    pdf.answer("We use LightGBM exclusively as a 'Feature Selector', not a predictor. By giving it all the data first, it outputs 'Importance Scores'. We strip away useless noisy parameters and only pass the 'Top 4' features (like pH, BOD) into our deep learning model. This drops computational time dramatically and prevents overfitting.")

    # Section 4: Accuracies & Error Math
    pdf.add_page()
    pdf.chapter_title("4. Mathematical Formulas & Errors (The Evaluator's Trap)")
    
    pdf.bullet_bold("Mean Squared Error (MSE)", "Our model achieved 0.0008")
    pdf.answer("Formula: MSE = (1/n) * SUM(Actual - Predicted)^2\nIt's the average of the squared errors. By squaring it, we eliminate negative signs and severely punish large prediction errors. An MSE of 0.0008 proves our DO predictions are remarkably tight.")
    
    pdf.bullet_bold("Mean Absolute Error (MAE)", "Our model achieved 0.0199")
    pdf.answer("Formula: MAE = (1/n) * SUM |Actual - Predicted|\nUnlike MSE, it doesn't exponentially punish outliers. It tells us that on average, our prediction is mathematically off by only ~0.02 mg/L from the real life sensor reading.")
    
    pdf.bullet_bold("Root Mean Squared Error (RMSE)", "Our model achieved 0.0285")
    pdf.answer("Formula: RMSE = SquareRoot [ (1/n) * SUM(Actual - Predicted)^2 ]\nMSE is mathematically squared (mg/L)^2, which isn't a real physical unit. RMSE applies a square root to bring the error metric perfectly back into standard (mg/L) units.")

    pdf.bullet_bold("R² (R-Squared)", "Our model achieved 96.28% (0.9628)")
    pdf.answer("Formula: R² = 1 - (SS_residual / SS_total)\nIn simple terms, R-Squared represents the percentage of exactly how much oxygen variance in the aquaculture our model can explain. 96.28% proves our LightGBM variables and Attention mechanism flawlessly capture what happens in the water.")

    # Section 5: The "Pro" Concepts
    pdf.chapter_title("5. Expert 'Pro-level' Viva Answers")
    
    pdf.question("What does the 'Attention Mechanism' literally do mathematically?")
    pdf.answer("In 24 hours of data, not all hours matter equally. Our custom Attention Layer mathematically scores each time step using a Tanh activation function, converts them to probabilities using Softmax, and assigns a 'Weight' multiplier. It tells the BiSRU: 'Mathematically ignore Hour 5, but multiply Hour 18 because Ammonia completely spiked'. It acts like human focus.")
    
    pdf.question("Did your neural network suffer from Overfitting? How did you stop it?")
    pdf.answer("Absolutely, so we established 'Early Stopping' and 'Validation Monitoring' (15% split). Overfitting happens when a model memorizes data instead of generalizing it. Our system checked Validation Loss every epoch. If training loss dropped but Validation Loss suddenly increased, Early Stopping triggered with a patience of 5, instantly halting training and restoring the best previous mathematical weights.")

    pdf.output('EXPERT_VIVA_PREPARATION_GUIDE.pdf')

if __name__ == "__main__":
    main()
