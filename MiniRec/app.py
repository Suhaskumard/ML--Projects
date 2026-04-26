from data_loader import load_data, create_matrix
from preprocessing import train_test_split_df
from model import user_similarity, item_similarity, predict_user_based
from recommender import get_top_n_recommendations
from evaluation import compute_rmse

def main():
    print("🚀 RecoSense Pro Running...\n")

    df = load_data()
    train_df, test_df = train_test_split_df(df)

    train_matrix = create_matrix(train_df)
    test_matrix = create_matrix(test_df)

    print("📊 Computing similarity...")
    user_sim = user_similarity(train_matrix)

    print("⚙️ Generating predictions...")
    predictions = predict_user_based(train_matrix.values, user_sim)

    print("📈 Evaluating model...")
    rmse = compute_rmse(predictions, train_matrix.values)
    print(f"RMSE: {rmse:.4f}\n")

    max_user_index = len(train_matrix) - 1

    while True:
        try:
            user_input = input(f"Enter user index (0 to {max_user_index}, or -1 to exit): ")
            
            # Handle non-integer input
            try:
                user_id = int(user_input)
            except ValueError:
                print("⚠️  Please enter a valid integer.\n")
                continue

            if user_id == -1:
                print("👋 Exiting RecoSense Pro. Goodbye!")
                break

            # Bounds checking
            if user_id < 0 or user_id > max_user_index:
                print(f"⚠️  User index out of range. Please enter a value between 0 and {max_user_index}.\n")
                continue

            recs = get_top_n_recommendations(user_id, train_matrix.values, predictions)

            print(f"\n🎯 Recommendations for User {user_id}:")
            
            if len(recs) == 0:
                print("No new recommendations available (user has rated all items).")
            else:
                for r in recs:
                    print(f"Item {train_matrix.columns[r]} (score: {predictions[user_id][r]:.2f})")

            print("\n" + "-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Exiting RecoSense Pro. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()

