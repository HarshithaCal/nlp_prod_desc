document.addEventListener('DOMContentLoaded', function() {
  const customizeButton = document.getElementById('customize');
  const personaSelect = document.getElementById('persona');
  const resultDiv = document.getElementById('result');
  const loader = document.getElementById('loader');
  const body = document.body;

  // Check if we're on an Amazon page
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    const activeTab = tabs[0];
    const isAmazon = activeTab.url.includes('amazon');
    
    if (!isAmazon) {
      resultDiv.innerHTML = '<p>Please navigate to an Amazon product page to use this extension.</p>';
      customizeButton.disabled = true;
    }
  });

  function setLoading(isLoading) {
    if (isLoading) {
      loader.style.display = 'block';
      customizeButton.disabled = true;
      body.classList.add('loading');
    } else {
      loader.style.display = 'none';
      customizeButton.disabled = false;
      body.classList.remove('loading');
    }
  }

  customizeButton.addEventListener('click', async function() {
    const selectedPersona = personaSelect.value;
    
    // Show loading state
    setLoading(true);
    
    try {
      // Query the active tab
      chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
        const activeTab = tabs[0];
        
        // Send message to the content script
        chrome.tabs.sendMessage(activeTab.id, {
          action: 'extractProductInfo',
          persona: selectedPersona
        }, function(response) {
          if (response && response.customDescription) {
            // Display the customized description
            resultDiv.innerHTML = `
              <h3>Customized Description:</h3>
              <p>${response.customDescription}</p>
            `;
          } else if (response && response.error) {
            // Display error message with specific handling for CAPTCHA
            if (response.error.includes("CAPTCHA")) {
              resultDiv.innerHTML = `
                <p>Error: ${response.error}</p>
                <p>Amazon is showing a security check. Please:</p>
                <ol>
                  <li>Solve the CAPTCHA in the main window</li>
                  <li>After you've solved it, navigate to the product page again</li>
                  <li>Try again by clicking "Customize Description"</li>
                </ol>
              `;
            } else {
              resultDiv.innerHTML = `<p>Error: ${response.error}</p>`;
            }
          } else {
            // Handle case where no response is received
            resultDiv.innerHTML = '<p>Failed to generate description. Make sure you are on an Amazon product page.</p>';
          }
        });
      });
    } catch (error) {
      resultDiv.innerHTML = '<p>Error: Could not generate description. Please try again.</p>';
    } finally {
      // Hide loading state
      setLoading(false);
    }
  });
}); 