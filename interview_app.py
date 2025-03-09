# Import modules
import os
import dspy
import streamlit as st
from dspy import Signature, Module, InputField, OutputField, Predict, Prediction, Example
from dspy.teleprompt import BootstrapFewShot
from llama_parse import LlamaParse
from dspy.evaluate import Evaluate



# Set environmental variables
os.environ["CO_API_KEY"] = "3xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxW"
cohere_api_key = os.environ["CO_API_KEY"]

# Configure LM
coh = dspy.Cohere(model='c4ai-aya-23')
dspy.settings.configure(lm=coh)

# Set up Resume and Job Description Parsing Functions
def parse_resume(resume_file):
    parser = LlamaParse(
        api_key="llx-ExxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxDK",
        result_type="text",
        verbose=True,
    )
    resume_document = parser.load_data(resume_file)
    resume_text = resume_document[0].text
    return resume_text
print("Resume parsed...")

def parse_job(job_file):
    parser = LlamaParse(
        api_key="llx-EXxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxGDK",
        result_type="text",
        verbose=True,
    )
    job_document = parser.load_data(job_file)
    job_text = job_document[0].text
    return job_text
print("Job parsed...")




def main():
    # Specify the paths to your files
    resume_file = "C:\\Users\\user\\Documents\\May 2024\\AI Eng\\Files\\Resume File1.pdf"
    job_file = "C:\\Users\\user\\Documents\\May 2024\\AI Eng\\Files\\Job File1.pdf"
    

    # Parse the resume and job description
    resume_text = parse_resume(resume_file)
    job_text = parse_job(job_file)
        
    return resume_text, job_text

if __name__ == "__main__":
    resume_text, job_text= main()

# Set up Question Generator module. Define "InterviewQuestion" signature with input fields for candidate skills and job requirements and an output field.

# Define Signatures
class GenerateInterviewQuestion(dspy.Signature):
    """Generate interview questions based on resume, job description, and last answer."""
    resume_text = dspy.InputField(desc="Text of the candidate's resume detailing their skills and experience")
    job_text = dspy.InputField(desc="Text of the job description outlining required qualifications")
    last_answer = dspy.InputField(desc="Text of the candidate's last answer", required=False)  # Make this optional if you want to start without an initial answer
    question = dspy.OutputField(desc="Interview question that relates to the candidate's suitability for the job based on their resume, job description, and last answer.")
    print("Signature updated with last answer...")

# Define Modules
class InterviewQuestionGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_question = Predict(GenerateInterviewQuestion)

    def forward(self, resume_text, job_text, last_answer=None):
        prediction = self.generate_question(resume_text=resume_text, job_text=job_text, last_answer=last_answer)
        return dspy.Prediction(question=prediction.question)
    print("Module updated to handle last answer...")


# Training Examples and Optimization
     
# Define examples with the necessary fields
    
train_example1 = dspy.Example(
    resume_text=resume_text, 
    job_text=job_text, 
    last_answer=" ", # To handle cases with no last answer
    previous_questions=[],
    question="Welcome to the Interview Phil. From the job text, this Senior Java developer role I am recruiting for requires you to have three key skills including Spring Boot for implementing Microservices architecture, J-unit for code texting, and SQL for database management. Do you have any or all of these three skills required for this position?")

train_example2 = dspy.Example(
    resume_text=resume_text, 
    job_text=job_text, 
    last_answer="Thank you for this interview opportunity. Yes, I have all the three skills required for this position. As you can see from my resume text, I gained experience with Spring Boot while working in Project HRMS Application at TrueLancer, I gained J-Unit experience while working at HTC-Global-Services, and I gained experience in SQL database management while working at Wipro Limited",
    previous_questions="Welcome to the Interview Phil. From the job text, this Senior Java developer role I am recruiting for requires you to have three key skills including Spring Boot for implementing Microservices architecture, J-unit for code texting, and SQL for database management. Do you have any or all of these three skills required for this position?",
    question="Okay, can you describe how you have used Spring Boot to ensure scalability and maintainability in microservices architecture in your previous role?")

train_example3 = dspy.Example(
    resume_text=resume_text, 
    job_text=job_text, 
    last_answer="In my previous role at TrueLancer where I was a senior software engineer, I designed microservices with Spring Boot, focusing on modular code and dynamic scaling, which significantly improved system maintainability and performance under varying loads.", 
    previous_questions="Okay, can you describe how you have used Spring Boot to ensure scalability and maintainability in microservices architecture in your previous role?",
    question="Okay, can you tell me about a specific project at TrueLancer where your modular coding skills really made a difference at the company?")

train_example4 = dspy.Example(
    resume_text=resume_text, 
    job_text=job_text, 
    last_answer="Regarding a specific project where I used my modular coding skills, I refactored TrueLancer's monolithic codebase into microservices which allowed us to independently update services, reduced deployment cycles from 5 days to 2 and accelerated deployment cycles.", 
    previous_questions="Okay, can you tell me about a specific project at TrueLancer where your modular coding skills really made a difference at the company?",
    question="Great. Could you delve deeper into how Spring Boot features, particularly its auto configuration feature supported this achievement of faster deployment cycles?")


print("Training examples created...")

# Tell DSPy about the specific input fields
trainset = [
    train_example1.with_inputs('resume_text', 'job_text', 'last_answer'),
    train_example2.with_inputs('resume_text', 'job_text', 'last_answer'),
    train_example3.with_inputs('resume_text', 'job_text', 'last_answer'),
    train_example4.with_inputs('resume_text', 'job_text', 'last_answer'),
]

print("Trainset created...")

# Set up Metric and Evaluation

class Assess(dspy.Signature):
  """Assesses the interview question for question count within a skill."""

  assessed_text = dspy.InputField()  # Interview question
  assessment_question = dspy.InputField()  # Question for LLM assessment
  assessment_answer = dspy.OutputField(desc="Yes or No")  # LLM's answer

skill_keywords = {
  "sql": ["sql"],
  "python": ["python"],
  "powerbi": ["powerbi"],
  }

def identify_current_skill(question, previous_questions):
  
  question_text = question.lower()
  for skill, keywords in skill_keywords.items():
    if any(keyword in question_text or any(keyword in q.lower() for q in previous_questions) for keyword in keywords):
      return skill
  return None

def metric(gold, pred, trace=None):
  # Identify current skill based on previous questions
  question = pred.question.lower()
  previous_questions = gold.previous_questions  # Access previous questions from training data
  current_skill = identify_current_skill(question, previous_questions)  # Use the identified skill

  # Configure LM (replace with your preferred LLM configuration)
  coh = dspy.Cohere(model='c4ai-aya-23')
  with dspy.context(lm=coh):
    question_count_check = f"Have there already been 3 questions asked about the same skill ({current_skill}) as this question?"
    question_count_check_result = dspy.Predict(Assess)(assessed_text=question, assessment_question=question_count_check)

  question_count_check_result = question_count_check_result.assessment_answer.lower() == 'yes'

  # Calculate score based on question count check
  score = 0 if question_count_check_result else 1  # Penalize for exceeding question count

  if trace is not None: return score == 1  # Strict check during compilation
  return score
print("Metric created...")


# Evaluate the model
evaluator = Evaluate(devset=trainset, num_threads=5, display_progress=True, display_table=11)
evaluation_score = evaluator(InterviewQuestionGenerator(), metric)
print("Evaluation is good...")
print(f"Average Metric: {evaluation_score}")

# Bootsrapping 
config = dict(max_bootstrapped_demos=5, max_labeled_demos=5)

# Set up teleprompter
teleprompter = BootstrapFewShot(metric=metric, **config)

compiled_module = teleprompter.compile(InterviewQuestionGenerator(), trainset=trainset)

print("Training completed...")


# Define a function to generate interview questions
def generate_question(resume_text, job_text, last_answer=None):
    pred = compiled_module(resume_text=resume_text, job_text=job_text, last_answer=last_answer)
    return pred.question

# Streamlit App
st.title("DSPy-Optimized AI Interview Assistant")
uploaded_resume = st.file_uploader("Upload your resume in PDF only:")
uploaded_job_desc = st.file_uploader("Upload your job description in PDF only")
generated_questions = []
generated_question_1 = None
parsed_resume_text = None 
parsed_job_desc_text = None

# Submit Button
submit_button = st.button("Generate Interview Question")

@st.cache_data
def parsee_resume(uploaded_resume):
    """
    Parses text from uploaded resume (PDF) using LlamaParse.

    Args:
        uploaded_resume (streamlit.uploaded_file.UploadedFile): Uploaded resume file.

    Returns:
        str: Extracted text from the resume, or None if parsing fails.
    """
    if uploaded_resume is None:
        return None
    try:
        # Save the uploaded file temporarily
        temp_path = os.path.join(os.getcwd(), "temp_resume.pdf")
        with open(temp_path, "wb") as temp_file:
            temp_file.write(uploaded_resume.read())

        # Initialize LlamaParse with your API key
        parser = LlamaParse(
            api_key="llx-EXxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxK",
            result_type="text",
            verbose=True,
        )

        # Parse the resume text
        with st.spinner("Parsing Resume..."):  # Added st.spinner
            resume_document = parser.load_data(temp_path)
            parsed_resume_text = resume_document[0].text

        # Clean up: remove the temporary file
        os.remove(temp_path)

        return parsed_resume_text
    except Exception as e:
        st.error(f"Error parsing resume with LlamaParse: {e}")
        return None

@st.cache_data
def parse_job_description(uploaded_job_desc):
    """
    Parses text from uploaded job description (PDF) using LlamaParse.

    Args:
        uploaded_job_desc (streamlit.uploaded_file.UploadedFile): Uploaded job description file.

    Returns:
        str: Extracted text from the job description, or None if parsing fails.
    """
    if uploaded_job_desc is None:
        return None
    try:
        # Save the uploaded file temporarily
        temp_path = os.path.join(os.getcwd(), "temp_job_desc.pdf")
        with open(temp_path, "wb") as temp_file:
            temp_file.write(uploaded_job_desc.read())

        # Initialize LlamaParse with your API key
        parser = LlamaParse(
            api_key="llx-EXxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxDK",
            result_type="text",
            verbose=True,
        )

        # Parse the job description text
        with st.spinner("Parsing Job Desc..."):  # Added st.spinner
            job_desc_document = parser.load_data(temp_path)
            parsed_job_desc_text = job_desc_document[0].text

        # Clean up: remove the temporary file
        os.remove(temp_path)

        return parsed_job_desc_text
    except Exception as e:
        st.error(f"Error parsing job description with LlamaParse: {e}")
        return None

# Initialize variables
generated_question = None
user_answer = None

# Check if both files are uploaded
if uploaded_resume and uploaded_job_desc:
    parsed_resume_text = parsee_resume(uploaded_resume)
    parsed_job_desc_text = parse_job_description(uploaded_job_desc)
    st.success("Both Uploaded and Parsed Successfully!")
else:
    st.error("Please upload both your resume and job description!")

# Generate the initial question
if parsed_resume_text and parsed_job_desc_text:
    generated_question = generate_question(parsed_resume_text, parsed_job_desc_text, "")
    if generated_question:
        st.write("**Generated Question:**")
        st.write(generated_question)

# Get the user's answer
answer = st.text_area("Your Answer here:")
submit_answer = st.button("Submit Your Answer")

# Process the answer (you can replace this with your actual processing logic)
if answer:
    user_answer = answer
    st.success("Answer submitted successfully!")

    # Generate the second question based on the user's answer
    second_question = generate_question(parsed_resume_text, parsed_job_desc_text, user_answer)
    if second_question:
        st.write("**Generated Second Question:**")
        st.write(second_question)
else:
    st.warning("Please provide an answer before proceeding.")







