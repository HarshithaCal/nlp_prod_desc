// Listen for messages from the popup
chrome.runtime.onMessage.addListener(function(request, sender, sendResponse) {
  console.log("Content script received message:", request);

  if (request.action === 'extractProductInfo') {
    try {
      // Check if we're on a CAPTCHA page
      const captchaElements = document.querySelectorAll('form[action*="captcha"]');
      const captchaText = document.body.textContent.includes("Type the characters you see in this image");
      
      if (captchaElements.length > 0 || captchaText) {
        sendResponse({ 
          error: "Amazon is showing a CAPTCHA page. Please solve the CAPTCHA and try again."
        });
        return true;
      }
      
      // Extract product information from the current page - this is the product description which we will use to generate a customized description.
      const productInfo = extractProductInfo();
      console.log("Extracted product info:", productInfo);
      
      if (!productInfo.title || !productInfo.description) {
        sendResponse({ 
          error: "Couldn't extract product information. Make sure you're on a product page."
        });
        return true;
      }
      
      // Send the product info to the Flask backend to generate the customized description
      fetch('http://localhost:5000/generate_description', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          product_name: productInfo.title,
          product_description: productInfo.description,
          persona: request.persona 
        })
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Something went wrong with the network');
        }
        return response.json();
      })
      .then(data => {
        // Send the custom description back to the popup
        console.log("Generated custom description:", data.custom_description);
        sendResponse({ customDescription: data.custom_description });
      })
      .catch(error => {
        console.error("Error sending request to backend:", error);
        sendResponse({ error: error.message });
      });
      
    } catch (error) {
      console.error("Error extracting product info:", error);
      sendResponse({ error: error.message });
    }
  }

  // Keep the message channel open for asynchronous response
  return true;
});

// Function to extract product information from the page - main function to extract product information.
function extractProductInfo() {
  // Initialization
  const productInfo = {
    title: '',
    description: '',
    price: null,
    brand: '',
    category: '',
    asin: '',
    images: []
  };

  // Check if we're on Amazon
  const isAmazon = window.location.hostname.includes('amazon');

  if (isAmazon) {
    // Amazon-specific extraction
    return extractAmazonProductInfo();
  }

  // Generic extraction for other sites
  // Try to extract title
  const titleSelectors = [
    'h1.product-title',
    'h1.product-name',
    'h1.product_title',
    'h1#productTitle',
    'h1.title',
    'h1'
  ];

  for (const selector of titleSelectors) {
    const titleElement = document.querySelector(selector);
    if (titleElement) {
      productInfo.title = titleElement.textContent.trim();
      break;
    }
  }

  // Try to extract description
  const descriptionSelectors = [
    'div.product-description',
    'div#productDescription',
    'div.description',
    'div.product-details',
    'p.product-description'
  ];

  for (const selector of descriptionSelectors) {
    const descElement = document.querySelector(selector);
    if (descElement) {
      productInfo.description = descElement.textContent.trim();
      break;
    }
  }

  return productInfo;
}

// Function to extract product information specifically from Amazon
function extractAmazonProductInfo() {
  console.log("Extracting Amazon product info...");
  
  const productInfo = {
    title: '',
    description: '',
    price: null,
    brand: '',
    category: '',
    asin: '',
    images: []
  };
  
  try {
    // Check if we can find any product information
    const pageContent = document.body.textContent;
    const isProductPage = (
      document.querySelector('#productTitle') ||
      document.querySelector('#title') ||
      document.querySelector('.product-title') ||
      document.querySelector('#aplus') ||
      document.querySelector('#feature-bullets')
    );
    
    if (!isProductPage) {
      console.warn("This doesn't appear to be a valid Amazon product page");
      return productInfo;
    }
    
    // Extract title - Amazon specific
    const titleSelectors = [
      '#productTitle', 
      '#title',
      '.product-title'
    ];
    
    for (const selector of titleSelectors) {
      const titleElement = document.querySelector(selector);
      if (titleElement) {
        productInfo.title = titleElement.textContent.trim();
        break;
      }
    }
    
    // Extract description
    // Try different selectors for product description on Amazon
    const descriptionSelectors = [
      '#productDescription',
      '#feature-bullets',
      '#aplus',
      '.a-expander-content'
    ];
    
    let descriptionText = '';
    
    for (const selector of descriptionSelectors) {
      const elements = document.querySelectorAll(selector);
      elements.forEach(element => {
        descriptionText += element.textContent.trim() + ' ';
      });
    }
    
    productInfo.description = descriptionText.trim();
    
    // If description is still empty, try to get at least the bullet points
    if (!productInfo.description) {
      const bulletPoints = document.querySelectorAll('#feature-bullets ul li');
      bulletPoints.forEach(bullet => {
        productInfo.description += bullet.textContent.trim() + ' ';
      });
    }
    
    // Try one more approach: product details section
    if (!productInfo.description) {
      const detailsElements = document.querySelectorAll('#detailBullets_feature_div li, #detailBulletsWrapper_feature_div li, #productDetails_feature_div tr');
      detailsElements.forEach(detail => {
        productInfo.description += detail.textContent.trim() + ' ';
      });
    }
    
    // Try to extract ASIN
    const asinMatch = window.location.pathname.match(/\/dp\/([A-Z0-9]{10})/);
    if (asinMatch && asinMatch[1]) {
      productInfo.asin = asinMatch[1];
    }
    
    return productInfo;
  } catch (error) {
    console.error("Error extracting Amazon product info:", error);
    return productInfo;
  }
}
