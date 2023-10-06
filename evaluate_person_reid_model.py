import torch
from sklearn.metrics import accuracy_score

# Load the trained model
model = SiameseNetwork(input_dim=feature_dim)
model.load_state_dict(torch.load("person_reid_model.pth"))

# Define a function to calculate Rank-1 Accuracy
def rank1_accuracy(query_features, gallery_features, query_labels, gallery_labels):
    # Implement the logic to calculate Rank-1 Accuracy here
    # Compare query features to gallery features and return accuracy

# Load your evaluation dataset (query and gallery sets) and labels here

# Extract features for query and gallery sets using the trained model

# Calculate Rank-1 Accuracy
rank1_acc = rank1_accuracy(query_features, gallery_features, query_labels, gallery_labels)

print(f"Rank-1 Accuracy: {rank1_acc}")
