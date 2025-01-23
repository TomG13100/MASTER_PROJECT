from train_keywords_model import train_keywords_model

def main():
    # Chemin des données et modèle
    csv_path = "data/keywords_labels.csv"
    model_save_path = "models/keywords_model.h5"

    # Entraîner le modèle
    train_keywords_model(csv_path, model_save_path)

if __name__ == "__main__":
    main()
