@echo off
echo Installing and upgrading dependencies from requirements.txt...
pip install --upgrade -r requirements.txt

echo Starting the PDF Q&A application with Gradio...
python app.py
pause