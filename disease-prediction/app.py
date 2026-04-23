from utils import predict_disease

print("Enter symptoms (1 = Yes, 0 = No)")

fever = int(input("Fever: "))
cough = int(input("Cough: "))
fatigue = int(input("Fatigue: "))

symptoms = [fever, cough, fatigue]

result = predict_disease(symptoms)

print(f"\nPredicted Disease: {result}")
