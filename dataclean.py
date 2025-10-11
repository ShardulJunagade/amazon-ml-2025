import pandas as pd
import re

# --- Step 1: Load CSV ---
df = pd.read_csv("dataset/train.csv")

# --- Step 2: Extract info from catalog_content ---
def parse_catalog(text):
    text = str(text)
    
    # Item Name
    item_name = re.search(r'Item Name:\s*(.*?)(?:\n|$)', text)
    item_name = item_name.group(1).strip() if item_name else None
    
    # Value & Unit
    value = re.search(r'Value:\s*([\d.]+)', text)
    value = value.group(1).strip() if value else None
    
    unit = re.search(r'Unit:\s*(.*?)(?:\n|$)', text)
    unit = unit.group(1).strip() if unit else None
    
    # Combine all bullet points
    bullets = re.findall(r'Bullet Point \d+:\s*(.*?)(?:\n|$)', text)
    description = ' '.join(bullets).strip() if bullets else None
    
    return pd.Series([item_name, description, value, unit])

# Apply parser
df[['Item Name', 'Description', 'Value', 'Unit']] = df['catalog_content'].apply(parse_catalog)

# Drop old column
df.drop(columns=['catalog_content'], inplace=True)

# --- Step 3: Clean and normalize units ---
df['Unit'] = df['Unit'].str.strip().str.lower().replace({
    'fl oz': 'oz',
    'ounce': 'oz',
    'ounces': 'oz',
    'count': 'count'
})

# --- Step 4: Reorder columns so price is last ---
cols = [c for c in df.columns if c != 'price'] + ['price']
df = df[cols]

# --- Step 5: Save final cleaned CSV ---
df.to_csv("train_cleaned.csv", index=False)
print("Cleaned data saved")