# Amazon Product Description Customizer

This Chrome extension customizes Amazon product descriptions based on user personas using the Tinyllama model. You can replace the model in the app.py file.
Check the comments in the app.py file.

# Setup Instructions

## 1. Set up the Flask Backend

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Flask server:
   ```
   python app.py
   ```
   This will start the server at http://localhost:5000

## 2. Chrome Extension setup

1. Open Chrome and go to "chrome://extensions/"
2. Enable "Developer mode" (toggle in the top-right corner)
3. Click "Load unpacked" and select the "code" folder containing the extension files
4. The extension should now be installed and visible in your Chrome toolbar


You can use any images for icons or create them using MS Paint like I did.

# Using the Extension

1. Go to any Amazon product page. Note: The extension works only on individual product pages (i.e., the page that appears after clicking on a specific product).
2. Click on the extension icon in your Chrome toolbar
3. Select a persona from the dropdown (any 1 out of 6)
4. Click "Customize Description"
5. The extension will parse the product information and send it to the code.
6. The customized description will be displayed in the popup after generation. Note as the description is extracted dynamically the model starts generating the custom description once you click generate so it might take a will to see the results so be patient. However, it shouldn't take too long as we are using tiny model.

## Troubleshooting

- Make sure the Flask backend is running at http://localhost:5000
- Check the browser console for any errors - very useful
- Ensure you're on an Amazon product page when using the extension (not on the home page. You need to click on any of the product to land on to the product page)