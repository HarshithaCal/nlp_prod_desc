from transformers import AutoModelForCausalLM, AutoTokenizer

# Load DistilGPT-2 model and tokenizer
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the persona and product description
persona = "tight budget, student, durability"
product_description = '''Sahara Sailor Water Bottle, 32oz Motivational Water Bottles with Time Marker, Sports Water Bottle with Silicone Spout and Straw, Tritan, BPA Free, Leakproof Bottle with Wide Mouth \
Brand: Sahara Sailor
Material: Tritan
Bottle Type: Silicone Spout with Removable Straw
Color: Bubble Gum
Capacity: 946.35 Milliliters

About this item:
Hydrating Over 15 Million Customers Since 2013: Founded in 2013, Sahara Sailor has sold over 15 million water bottles worldwide, becoming a trusted name in hydration. Our commitment to quality, innovation, and customer satisfaction has made us a favorite among active individuals, fitness enthusiasts, and anyone looking to stay hydrated. Join our global community and discover why millions rely on Sahara Sailor for their hydration needs.
Commitment to Sustainability: Sahara Sailor is a steadfast partner of Climate Neutral, dedicated to reducing our carbon footprint. We also collaborate with One Tree Planted to plant trees, helping to restore and protect our planet's forests. Our commitment to sustainability ensures that every purchase you make supports a healthier, greener world. Join us in our mission to make a positive impact on the environment.
Dual-Use Lid: This water bottle with times to drink features an upgraded lid with a soft silicone spout, great for direct drinking. Additionally, it includes a straw, allowing you to effortlessly sip your beverage with ease. This versatile lid design ensures you stay hydrated in a very convenient way possible.
Motivational Quotes with Time Marker: This sports water bottle features time markers and motivational quotes for each hydration milestone you reach. It's designed to inspire you and provide a sense of accomplishment, helping you develop and maintain the healthy habit of staying hydrated. Stay motivated and enjoy a fulfilling hydration journey with every sip.
BPA-Free Tritan Material: This water bottle is crafted from Tritan material, ensuring safety for both you and the planet. Durable and BPA-free, it offers a reliable and sustainable option for your hydration needs. Enjoy peace of mind knowing that you're making a healthy and green choice with every use.
'''

# Construct the input prompt
input_prompt = f"Generate customized product description to convince the persona to buy a product by changing the generic product description to a customized product description.\
    Persona: {persona}.\
    Product description: {product_description}. \
    Customized product description: "

# Tokenize the input prompt
inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
input_ids = inputs['input_ids']
attention_mask = inputs["attention_mask"]

# Generate the output
output = model.generate(input_ids,
                        attention_mask=attention_mask,
                        num_return_sequences=1,  # Generate 1 unique output
                        max_length=1024,         # Max length defined for this model is 1024
                        top_k=50,
                        top_p=0.95,
                        do_sample=True,
                        temperature=0.7,         # Controls randomness: <1 for deterministic results, >1 for more creative outputs
                        truncation=True,
                        pad_token_id=tokenizer.eos_token_id  # Use EOS token for padding since DistilGPT-2 doesn't have a pad token
                        )

# Decode and print the generated text
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
