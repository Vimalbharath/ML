from google.colab import drive
import shutil

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Copy the file directly to your Drive
shutil.copy('log/model_00049.pt', '/content/drive/MyDrive/model_00049.pt')
print("Successfully copied weights.pt to Google Drive!")