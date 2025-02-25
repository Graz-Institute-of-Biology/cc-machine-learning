import torch

model_path = "C:\\Users\\faulhamm\\Documents\\Philipp\\Code\\cc-machine-learning\\results\\03-cc-graz\\exp_g_mit_b1_20_08ca64\\best_model.pth"

print(model_path)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
print("Checkpoint loaded!")
print("Summary: ")
print("Epoch: ", checkpoint['epoch'])
print("Ontology: ", checkpoint['ontology'])
print("Model loaded!")