# Advanced AI Academic Assistant 🎓

An intelligent academic companion powered by Google's Gemini AI that helps students and educators with data visualization, mathematical analysis, and assignment processing.

## Features 🌟

- **Interactive Data Visualization** 📊
  - Create custom visualizations using natural language descriptions
  - Support for multiple data input methods (CSV upload, text input, sample datasets)
  - Export visualizations as Jupyter notebooks
  - Interactive Plotly charts and static Matplotlib/Seaborn plots

- **Advanced Mathematical Analysis** 🧮
  - Text and image-based math problem solving
  - LaTeX and plain text notation support
  - Step-by-step solution explanations
  - OCR capability for handwritten math problems

- **Smart Assignment Processing** 📝
  - Subject-specific analysis
  - Detailed solution strategies
  - Practice problem suggestions
  - Assessment criteria and common pitfalls

- **Progress Tracking & History** 💾
  - Track study progress
  - Save favorite solutions
  - Export study history
  - Filter and search past work

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/Saoud30/academic-assistant.git
cd academic-assistant
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

## Usage 🎯

1. Start the application:
```bash
streamlit run app.py
```

2. Navigate to the provided local URL (typically `http://localhost:8501`)

3. Choose a feature from the sidebar navigation:
   - Data Visualization
   - Math Helper
   - Assignment Analyzer
   - Study History
   - Settings

## Requirements 📋

See `requirements.txt` for a complete list of dependencies. Key requirements include:

- Python 3.8+
- Streamlit
- Google Generative AI
- Pandas
- Matplotlib
- Seaborn
- Plotly
- PyTesseract
- OpenCV
- SymPy
- latex2sympy2

## Environment Setup ⚙️

1. Install Tesseract OCR:
   - **Windows**: Download and install from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

2. Ensure Tesseract is in your system PATH

## Contributing 🤝

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License 📄

MIT License - see LICENSE file for details

## Acknowledgments 👏

- Google Generative AI for powering the assistant
- Streamlit for the web interface
- Open-source community for various libraries used

---

