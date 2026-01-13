# Fonts for Share Card Generator

This directory contains fonts used by the share card generator to create beautiful social media cards.

## Required Fonts

### 1. Playfair Display (Serif - for titles)
- **Download from**: https://fonts.google.com/specimen/Playfair+Display
- **Required files**:
  - `PlayfairDisplay-Bold.ttf`
  - `PlayfairDisplay-Regular.ttf`

### 2. Inter (Sans-serif - for body text)
- **Download from**: https://fonts.google.com/specimen/Inter
- **Required files**:
  - `Inter-Regular.ttf`
  - `Inter-Bold.ttf`

## Installation Instructions

### Option 1: Download Manually
1. Visit the Google Fonts links above
2. Click "Download family" button
3. Extract the TTF files from the downloaded ZIP
4. Copy the required TTF files to this directory

### Option 2: Using wget (if available)
```bash
# Note: This downloads from Google Fonts API
# You may need to extract and rename files manually
cd backend/fonts

# Playfair Display
wget "https://fonts.google.com/download?family=Playfair%20Display" -O playfair.zip
unzip playfair.zip "*.ttf" -d playfair
cp playfair/static/PlayfairDisplay-Bold.ttf .
cp playfair/static/PlayfairDisplay-Regular.ttf .
rm -rf playfair playfair.zip

# Inter
wget "https://fonts.google.com/download?family=Inter" -O inter.zip
unzip inter.zip "*.ttf" -d inter
cp inter/static/Inter_24pt-Regular.ttf ./Inter-Regular.ttf
cp inter/static/Inter_24pt-Bold.ttf ./Inter-Bold.ttf
rm -rf inter inter.zip
```

## Fallback Behavior

If fonts are not found, the share card generator will automatically fall back to the default PIL font. However, for the best visual quality, please ensure all fonts are properly installed.

## Verification

After installation, this directory should contain:
```
backend/fonts/
├── PlayfairDisplay-Bold.ttf
├── PlayfairDisplay-Regular.ttf
├── Inter-Regular.ttf
├── Inter-Bold.ttf
└── README.md
```

You can verify the fonts are loaded correctly by running:
```python
from PIL import ImageFont
font = ImageFont.truetype('backend/fonts/PlayfairDisplay-Bold.ttf', 60)
print("Fonts loaded successfully!")
```
