AI Art Detector\
A deep learning model that classifies uploaded artwork images as AI-geenrated or human created using a fine tuned CNN 

Overview\
This project uses a PyTorch-based neural network (ResNet-18) to perform binary classification(AI vs. human) on uploaded artwork. It includes dataloading, training, evaluation, and inference pipelines

Install dependencies from requirements.txt to run

Future Improvements\
The current prototype indicates overfitting due to limited access to training data. This will be addressed in the final model by training the model with the full dataset and improving data augmentation strategies. Additionally, a frontend web applications will be implemented to deploy the complete model for users to upload and receive real-time feedback.
