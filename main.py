"""CLI entry point for the March Madness GNN predictor."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="NCAA March Madness GNN Predictor")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the GNN model")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    train_parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension size")
    train_parser.add_argument("--save-path", type=str, default="model.pt", help="Path to save trained model")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict tournament bracket")
    predict_parser.add_argument("--model-path", type=str, default="model.pt", help="Path to trained model")

    args = parser.parse_args()

    if args.command == "train":
        from src.train import train
        train(
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            save_path=args.save_path,
        )
    elif args.command == "predict":
        from src.predict import predict_bracket
        predict_bracket(model_path=args.model_path)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
