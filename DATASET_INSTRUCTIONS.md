# Dataset Download Instructions

## Recipe Generation Dataset (recipe_nlg)

The recipe generation model requires the Recipe NLG dataset which must be manually downloaded:

1. **Download the dataset**:
   ```bash
   # Download the dataset (if not already done)
   mkdir -p ~/Downloads/recipe_data
   cd ~/Downloads
   wget https://recipenlg.blob.core.windows.net/recipenlg-misc/dataset.zip
   unzip dataset.zip -d ~/Downloads/recipe_data
   ```

2. **Prepare the dataset**:
   ```bash
   # Create a directory for the dataset
   mkdir -p ~/manual_data
   
   # Move the dataset file to the directory
   cp ~/Downloads/recipe_data/full_dataset.csv ~/manual_data/
   ```

3. **Use the dataset**:
   ```bash
   # Specify the data directory when running commands
   python -m src.data.recipe_prepare_dataset --data_dir ~/manual_data
   
   # Or use the make command with the DATA_DIR variable
   make recipe-data DATA_DIR=~/manual_data
   ```

## Sentiment Analysis Dataset (imdb)

The sentiment analysis model uses the IMDB dataset which is automatically downloaded from Hugging Face, so no manual steps are required.
