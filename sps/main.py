import os
import subprocess

print("🔥 SPS - Advanced Human Activity Recognition System")
print("=" * 50)
print("1. 🚀 Train LSTM Model")
print("2. 🔮 Predict Activities") 
print("3. 📊 View Project Info")
print("4. ❌ Exit")

while True:
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n🚀 Training model...")
        os.system("python train.py")
    elif choice == "2":
        print("\n🔮 Making predictions...")
        os.system("python predict.py")
    elif choice == "3":
        print("\n📁 Project structure ready!")
        os.system("tree .") if os.name == "nt" else os.system("ls -la")
        print("\n📖 See README.md for full docs")
    elif choice == "4":
        print("👋 Goodbye!")
        break
    else:
        print("❌ Invalid choice! Try 1-4")

