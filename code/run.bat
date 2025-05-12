@echo off
echo Installing requirements...
pip install -r requirements.txt
echo Running TinyLlama fine-tuning...
python finetune_tinyllama.py
echo Finetuned TinyLlama model
echo Running Gemma model...
python gemma-3-4b-it.py
echo Finetuned Gemma 3 4B model
echo Running comparison...
python comparison.py
echo Comparison done

pause 