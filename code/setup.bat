@echo off
echo This setup is for guidance on how to set up the extension. Alternatively, you can install the requirements.txt file using pip install -r requirements.txt and refer to the README.md file for more details.
echo Or you just want to run the code without using the extension then use the python model_name.py file

echo Setting up Amazon Product Description Customizer...
echo Launch this setup.bat file from the code folder else the paths will be incorrect.
echo Installing required Python packages...
pip install -r requirements.txt
pip install pillow

echo Setup complete
echo Follow these steps to use the extension:
echo 1. Start the Flask backend by running: python app.py
echo 2. Load the extension in Chrome:
echo    - Go to chrome://extensions/
echo    - Enable "Developer mode"
echo    - Click "Load unpacked" and select this folder
echo 3. Go to an Amazon product page and click on the extension icon. Not that this extension only works on Amazon product pages so select one product from Amazon.
pause 