import os
import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from PIL import Image
import io
import pytesseract
import nbformat
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import sympy as sp
from latex2sympy2 import latex2sympy
import cv2
from typing import List, Dict, Any, Optional, Union

class AcademicAssistant:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = self.configure_genai()
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'history' not in st.session_state:
            st.session_state.history = []
        if 'code' not in st.session_state:
            st.session_state.code = None
        if 'favorites' not in st.session_state:
            st.session_state.favorites = []
        if 'settings' not in st.session_state:
            st.session_state.settings = {
                'theme': 'light',
                'math_notation': 'latex',
                'code_style': 'default'
            }

    def configure_genai(self) -> Optional[Any]:
        """Configure Gemini AI with error handling and model selection"""
        try:
            genai.configure(api_key=self.api_key)

            # Set up generation configuration
            generation_config = {
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }

            # Initialize model with correct name and configuration
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config
            )

            return model
        except Exception as e:
            st.error(f"Error configuring Gemini AI: {str(e)}")
            return None

    def process_image(self, image: Image.Image) -> Image.Image:
        """Enhanced image processing with OpenCV"""
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        blurred = cv2.GaussianBlur(opencv_image, (3, 3), 0)  # Reduced kernel size

        # Apply image preprocessing
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        black_pixels = np.sum(thresh == 0)
        total_pixels = thresh.size
        black_pixel_ratio = black_pixels / total_pixels
        #print(f"Black pixel ratio: {black_pixel_ratio}") #uncomment to check black pixel ratio during testing
        if black_pixel_ratio < 0.1 or black_pixel_ratio > 0.9:
           st.warning("Thresholding might be ineffective. Check image quality.")


        # Convert back to PIL Image
        return Image.fromarray(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))

    def extract_text_from_images(self, images: List[Image.Image]) -> List[str]:
        """Enhanced OCR with better preprocessing and error handling"""
        extracted_text = []
        for image in images:
            try:
                # Process image
                processed_image = self.process_image(image)

                # Perform OCR with custom configuration
                custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=()abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
                text = pytesseract.image_to_string(processed_image, config=custom_config)

                # Clean extracted text
                text = self.clean_extracted_text(text)
                extracted_text.append(text)
            except Exception as e:
                st.error(f"Error extracting text from image: {str(e)}")
                extracted_text.append("")
        return extracted_text

    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove unnecessary whitespace and normalize characters
        text = ' '.join(text.split())
        text = text.replace('—', '-').replace('’', "'").replace('"', '"').replace('"', '"')
        return text

    def create_jupyter_notebook(self, content: Dict[str, Any]) -> Any:
        """Create enhanced Jupyter notebook with markdown cells and metadata"""
        try:
            # Create a new notebook using nbformat
            nb = {
                "nbformat": 4,
                "nbformat_minor": 4,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python 3",
                        "language": "python",
                        "name": "python3"
                    },
                    "language_info": {
                        "codemirror_mode": {
                            "name": "ipython",
                            "version": 3
                        },
                        "file_extension": ".py",
                        "mimetype": "text/x-python",
                        "name": "python",
                        "nbconvert_exporter": "python",
                        "pygments_lexer": "ipython3",
                        "version": "3.8.0"
                    }
                },
                "cells": []
            }

            # Add title and description
            nb["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": f"# {content['title']}\nGenerated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            })

            # Add content cells
            for item in content['cells']:
                if item['type'] == 'code':
                    cell = {
                        "cell_type": "code",
                        "metadata": {},
                        "execution_count": None,
                        "outputs": [],
                        "source": item['content']
                    }
                else:
                    cell = {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": item['content']
                    }
                nb["cells"].append(cell)

            return nb
        except Exception as e:
            st.error(f"Error creating Jupyter notebook: {str(e)}")
            return None

    def generate_visualization(self, prompt: str, data: Optional[pd.DataFrame] = None, include_comments: bool = True) -> Dict[str, Any]:
        """Enhanced visualization generation that works with or without input data"""
        try:
            # Adjust the system prompt based on whether data is provided
            comment_instruction = "Include detailed comments explaining the code" if include_comments else "Generate code without comments"

            if data is None:
                system_prompt = f"""You are a data visualization expert. Generate complete, executable Python code that:
                1. Creates sample data within the code if needed
                2. Includes all necessary imports
                3. Creates the requested visualization
                4. {comment_instruction}
                5. Returns the figure object
                6. Uses either Plotly for interactive or Matplotlib/Seaborn for static visualizations
                7. Wraps everything in a function named 'create_visualization' that takes no parameters

                Example format:
                import numpy as np
                import matplotlib.pyplot as plt

                def create_visualization():
                    # Generate sample data
                    x = np.linspace(0, 10, 100)
                    y = np.sin(x)

                    # Create visualization
                    fig, ax = plt.subplots()
                    ax.plot(x, y)
                    return fig
                """
            else:
                system_prompt = f"""You are a data visualization expert. Generate complete, executable Python code that:
                1. Uses the provided DataFrame with columns: {', '.join(data.columns)}
                2. Includes all necessary imports
                3. Creates the requested visualization
                4. {comment_instruction}
                5. Returns the figure object
                6. Uses either Plotly for interactive or Matplotlib/Seaborn for static visualizations
                7. Wraps everything in a function named 'create_visualization' that takes a DataFrame parameter

                Example format:
                import plotly.express as px

                def create_visualization(df):
                    fig = px.scatter(df, x='column1', y='column2')
                    return fig
                """

            chat = self.model.start_chat(history=[])
            response = chat.send_message([system_prompt, prompt])

            # Extract and clean the code
            code = response.text.strip()

            # Remove any markdown code block indicators
            code = code.replace('```python', '').replace('```', '').strip()

            # Validate the code structure
            if 'def create_visualization' not in code:
                raise ValueError("Generated code must contain a 'create_visualization' function")

            # Basic code validation
            try:
                compile(code, '<string>', 'exec')
            except SyntaxError as e:
                lines = code.split('\n')
                error_line = e.lineno - 1 if e.lineno <= len(lines) else len(lines) - 1
                context = '\n'.join(lines[max(0, error_line-2):min(len(lines), error_line+3)])

                return {
                    'error': f"Syntax error at line {e.lineno}: {str(e)}\nContext:\n{context}",
                    'code': code,
                    'library': 'unknown'
                }

            # Detect visualization library
            library = 'unknown'
            if 'plotly' in code:
                library = 'plotly'
            elif 'matplotlib' in code or 'seaborn' in code:
                library = 'matplotlib'

            return {
                'code': code,
                'library': library,
                'error': None
            }

        except Exception as e:
            return {
                'error': f"Error generating visualization: {str(e)}",
                'code': None,
                'library': 'unknown'
            }

    def solve_math_problem(self, problem: str, notation: str = 'latex') -> Dict[str, Any]:
        """Enhanced math problem solver with symbolic computation and OCR support"""
        try:
            if notation == 'latex':
                try:
                    symbolic_expr = latex2sympy(problem)
                    problem = str(symbolic_expr)
                except:
                    pass

            system_prompt = """You are an advanced mathematics tutor. Provide:
            1. Problem classification and approach
            2. Detailed step-by-step solution with explanations
            3. Alternative solution methods if applicable
            4. Verification and practical applications
            5. Common mistakes to avoid"""

            chat = self.model.start_chat(history=[])
            response = chat.send_message([system_prompt, problem])

            return {
                'solution': response.text,
                'symbolic': str(symbolic_expr) if 'symbolic_expr' in locals() else None,
                'latex': problem if notation == 'latex' else None
            }
        except Exception as e:
            st.error(f"Error solving math problem: {str(e)}")
            return None

    def analyze_assignment(self, text: str, subject: str) -> Dict[str, Any]:
        """Enhanced assignment analyzer with subject-specific processing"""
        try:
            system_prompt = f"""You are an expert {subject} tutor. Provide:
            1. Problem breakdown and key concepts
            2. Detailed solution strategy
            3. Step-by-step implementation
            4. Visual aids or diagrams where helpful
            5. Practice problems and additional resources
            6. Assessment criteria and common pitfalls"""

            chat = self.model.start_chat(history=[])
            response = chat.send_message([system_prompt, text])

            return {
                'analysis': response.text,
                'subject': subject,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Error analyzing assignment: {str(e)}")
            return None

def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a sample dataset for demonstration"""
    if name == "Iris":
        return sns.load_dataset("iris")
    elif name == "Titanic":
        return sns.load_dataset("titanic")
    elif name == "Housing":
        return pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/Housing.csv")
    else:
        raise ValueError(f"Unknown dataset: {name}")

def main():
    st.set_page_config(page_title="Advanced AI Academic Assistant", layout="wide")

    # Get API key from environment variable

    if 'GEMINI_API_KEY' not in st.secrets:
        st.error("Please set up your GEMINI_API_KEY in the Streamlit secrets")
        st.info("To set up secrets in Streamlit Cloud:")
        st.code("""
1. Go to your app dashboard
2. Click on 'Settings' ⚙️
3. Go to 'Secrets' section
4. Add your secret like this:
GEMINI_API_KEY='your-api-key-here'
        """)
        return

    # Initialize the assistant
    assistant = AcademicAssistant(st.secrets["GEMINI_API_KEY"])

    # Custom CSS for better UI
    st.markdown("""
        <style>
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        .success-message {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #d4edda;
            border-color: #c3e6cb;
            color: #155724;
        }
        </style>
    """, unsafe_allow_html=True)

    # Main title and description
    st.title("🎓 Advanced AI Academic Assistant")
    st.markdown("""
    An intelligent academic companion featuring:
    - 📊 Interactive Data Visualization
    - 🧮 Advanced Mathematical Analysis
    - 📝 Smart Assignment Processing
    - 📚 Subject-Specific Tutoring
    - 💾 Progress Tracking & History
    """)

    # Sidebar navigation and settings
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a feature",
            ["Data Visualization", "Math Helper", "Assignment Analyzer", "Study History", "Settings"]
        )

        st.header("Settings")
        theme = st.selectbox("Theme", ["Light", "Dark"], key="sidebar_theme_select")
        math_notation = st.selectbox("Math Notation", ["LaTeX", "Plain Text"], key="sidebar_notation_select")

    # Main content area based on selected page
    if page == "Data Visualization":
        st.header("📊 Interactive Data Visualization Studio")

        # Data input options
        data_input = st.radio("Choose data input method:", ["Upload CSV", "Paste Data", "Sample Dataset"], key="data_input_radio")

        if data_input == "Upload CSV":
            uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                st.dataframe(df.head())
                st.session_state.current_df = df
        elif data_input == "Paste Data":
            data_text = st.text_area("Paste your data (CSV format)", key="data_text_area")
            if data_text:
                df = pd.read_csv(StringIO(data_text))
                st.dataframe(df.head())
                st.session_state.current_df = df
        else:
            dataset = st.selectbox("Choose a sample dataset", ["Iris", "Titanic", "Housing"], key="dataset_select")
            df = load_sample_dataset(dataset)
            st.dataframe(df.head())
            st.session_state.current_df = df

        # Code generation options
        col1, col2 = st.columns(2)
        with col1:
            include_comments = st.checkbox("Include code comments", value=True, key="comment_checkbox")
        with col2:
            show_visualization = st.checkbox("Show visualization", value=True, key="viz_checkbox")

        viz_prompt = st.text_area(
            "Describe the visualization you want to create:",
            height=100,
            placeholder="Example: Create an interactive scatter plot showing the relationship between sepal length and sepal width, colored by species",
            key="viz_prompt_area"
        )

        def create_notebook_content(code: str, viz_prompt: str, data_input: str, dataset_name: str = None) -> dict:
            """Create Jupyter notebook content with visualization code"""
            # Import statements cell
            imports = """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np"""

            # Data loading cell based on input method
            if data_input == "Sample Dataset":
                data_loading = f"""# Load sample dataset
from seaborn import load_dataset
df = load_dataset("{dataset_name.lower()}")"""
            else:
                data_loading = """# Load your data
# Replace 'your_data.csv' with your actual data file
df = pd.read_csv('your_data.csv')"""

            # Create notebook structure
            cells = [
                nbformat.v4.new_markdown_cell(f"# Visualization Generator\n\nPrompt: {viz_prompt}"),
                nbformat.v4.new_code_cell(imports),
                nbformat.v4.new_code_cell(data_loading),
                nbformat.v4.new_code_cell(code)
            ]

            # Create notebook
            nb = nbformat.v4.new_notebook()
            nb.cells = cells
            return nb

        if st.button("Generate Visualization", key="viz_button"):
            if viz_prompt:
                with st.spinner("Creating visualization..." if show_visualization else "Generating code..."):
                    # Pass the DataFrame only if data is loaded
                    current_df = st.session_state.get('current_df')
                    viz_result = assistant.generate_visualization(
                        viz_prompt,
                        current_df if current_df is not None else None,
                        include_comments
                    )

                    if viz_result:
                        if viz_result.get('error'):
                            st.error(viz_result['error'])
                        else:
                            # Display code in collapsible section
                            with st.expander("View Generated Code"):
                                st.code(viz_result['code'], language='python')

                            # Create download button for Jupyter notebook
                            dataset_name = st.session_state.get('dataset_select') if data_input == "Sample Dataset" else None
                            notebook = create_notebook_content(
                                viz_result['code'],
                                viz_prompt,
                                data_input,
                                dataset_name
                            )

                            # Convert notebook to JSON string
                            notebook_json = nbformat.writes(notebook)

                            # Create download button
                            st.download_button(
                                label="📥 Download as Jupyter Notebook",
                                data=notebook_json,
                                file_name="visualization.ipynb",
                                mime="application/x-ipynb+json",
                                key="notebook_download"
                            )

                        # Execute visualization code only if show_visualization is True
                        if show_visualization:
                            try:
                                # Create a local namespace for execution
                                local_vars = {
                                    'pd': pd,
                                    'plt': plt,
                                    'sns': sns,
                                    'px': px,
                                    'go': go,
                                    'np': np
                                }

                                # Add DataFrame to local_vars if it exists
                                if current_df is not None:
                                    local_vars['df'] = current_df

                                # Execute the code
                                exec(viz_result['code'], globals(), local_vars)

                                # Get the visualization function
                                if 'create_visualization' in local_vars:
                                    # Call the function with or without DataFrame
                                    if current_df is not None:
                                        fig = local_vars['create_visualization'](current_df)
                                    else:
                                        fig = local_vars['create_visualization']()

                                    # Display the figure
                                    if viz_result['library'] == 'plotly':
                                        st.plotly_chart(fig)
                                    else:
                                        st.pyplot(fig)
                                        plt.close()
                                else:
                                    st.error("Visualization function not found in generated code")
                            except Exception as e:
                                st.error(f"Error executing visualization: {str(e)}")
                                st.code(viz_result['code'], language='python')

                        # Add to history
                        st.session_state.history.append({
                            "type": "visualization",
                            "prompt": viz_prompt,
                            "code": viz_result['code'],
                            "timestamp": datetime.now().isoformat()
                        })
            else:
                st.warning("Please enter a description of the visualization you want to create.")

    elif page == "Math Helper":
        st.header("🧮 Advanced Mathematical Analysis")

        input_method = st.radio("Input Method:", ["Text", "Image"], key="math_input_method")

        if input_method == "Text":
            math_problem = st.text_area("Enter your math problem:", height=100, key="math_problem_text")
        else:  # Image input
            uploaded_images = st.file_uploader("Upload image(s) of your math problem", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="math_problem_images")
            if uploaded_images:
                extracted_texts = []
                for i, uploaded_image in enumerate(uploaded_images):
                    image = Image.open(uploaded_image)
                    extracted_text = assistant.extract_text_from_images([image])[0]
                    extracted_texts.append(extracted_text)

                    st.subheader(f"Extracted Text from Image {i+1}")
                    editable_text = st.text_area(f"Edit the extracted text if needed:", value=extracted_text, key=f"extracted_text_{i}")  # Make text editable
                    extracted_texts[i] = editable_text  # Update with edited text

                math_problem = "\n".join(extracted_texts) # Join extracted texts

                st.subheader("Combined Extracted Text (will be used for solving)")
                st.text(math_problem)

        notation = st.radio("Input notation (for text input):", ["LaTeX", "Plain Text"], key="notation_radio")

        if st.button("Solve Problem", key="solve_button"):
            if math_problem:
                with st.spinner("Solving..."):
                    solution = assistant.solve_math_problem(math_problem, notation.lower() if input_method == "Text" else "plain") # Default to plain text for image input
                    if solution:
                        st.markdown("### Solution")
                        st.write(solution['solution'])
                        if solution['symbolic']:
                            st.latex(solution['symbolic'])

    elif page == "Assignment Analyzer":
        st.header("📝 Smart Assignment Analysis")
        subject = st.selectbox("Select subject:", ["Mathematics", "Physics", "Chemistry", "Computer Science", "Other"], key="subject_select")
        assignment_text = st.text_area(
            "Enter your assignment text:",
            height=200,
            placeholder="Paste your assignment question or problem here...",
            key="assignment_text_area"
        )

        if st.button("Analyze Assignment", key="analyze_button"):
            if assignment_text:
                with st.spinner("Analyzing your assignment..."):
                    analysis = assistant.analyze_assignment(assignment_text, subject)
                    if analysis:
                        st.markdown("### Analysis Results")
                        st.write(analysis['analysis'])

                        # Add to history
                        st.session_state.history.append({
                            "type": "assignment",
                            "subject": subject,
                            "text": assignment_text,
                            "analysis": analysis['analysis'],
                            "timestamp": analysis['timestamp']
                        })

    elif page == "Study History":
        st.header("📚 Study History & Progress Tracking")

        if not st.session_state.history:
            st.info("No study history available yet. Start using the assistant to build your history!")
        else:
            # Filter options
            filter_type = st.selectbox(
                "Filter by type:",
                ["All", "visualization", "math", "assignment"]
            )

            filtered_history = st.session_state.history
            if filter_type != "All":
                filtered_history = [item for item in filtered_history if item["type"] == filter_type]

            # Display history items
            for item in filtered_history:
                with st.expander(f"{item['type'].title()} - {item['timestamp']}"):
                    if item['type'] == 'visualization':
                        st.write("Visualization Prompt:", item['prompt'])
                        st.code(item['code'], language='python')
                    elif item['type'] == 'assignment':
                        st.write("Subject:", item['subject'])
                        st.write("Assignment Text:", item['text'])
                        st.write("Analysis:", item['analysis'])

                    # Add favorite/bookmark functionality
                    if st.button("⭐ Favorite", key=f"fav_{item['timestamp']}"):
                        if item not in st.session_state.favorites:
                            st.session_state.favorites.append(item)
                            st.success("Added to favorites!")

    elif page == "Settings":
        st.header("⚙️ Settings & Preferences")

        # Theme settings
        st.subheader("Appearance")
        new_theme = st.selectbox(
            "Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.settings['theme'] == 'light' else 1,
            key="settings_theme_select"
        )
        if new_theme.lower() != st.session_state.settings['theme']:
            st.session_state.settings['theme'] = new_theme.lower()
            st.success("Theme updated! Please refresh the page to see changes.")

        # Math notation settings
        st.subheader("Math Settings")
        new_notation = st.selectbox(
            "Default Math Notation",
            ["LaTeX", "Plain Text"],
            index=0 if st.session_state.settings['math_notation'] == 'latex' else 1,
            key="settings_notation_select"
        )
        if new_notation.lower() != st.session_state.settings['math_notation']:
            st.session_state.settings['math_notation'] = new_notation.lower()
            st.success("Math notation preference updated!")

        # Code style settings
        st.subheader("Code Settings")
        new_code_style = st.selectbox(
            "Code Display Style",
            ["Default", "Monokai", "Solarized"],
            index=["default", "monokai", "solarized"].index(st.session_state.settings['code_style']),
            key="settings_code_style_select"
        )
        if new_code_style.lower() != st.session_state.settings['code_style']:
            st.session_state.settings['code_style'] = new_code_style.lower()
            st.success("Code style updated!")

        # Data export
        st.subheader("Data Management")
        if st.button("Export Study History", key="export_button"):
            history_data = {
                "history": st.session_state.history,
                "favorites": st.session_state.favorites,
                "settings": st.session_state.settings,
                "export_date": datetime.now().isoformat()
            }

            # Convert to JSON string
            json_str = json.dumps(history_data, indent=2)

            # Create download button
            st.download_button(
                label="Download History",
                data=json_str,
                file_name="study_history.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
