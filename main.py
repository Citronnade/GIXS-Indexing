import argparse
import deep_d
import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Powder diffraction indexer")
    #   parser.add_argument("--optimizer", help="type of optimizer to use.  Options: Adam", choices=["Adam"])
    parser.add_argument("--batch_size", type=int, help="batch size to use in training")
    parser.add_argument("--num_epochs", type=int, help="number of epochs to train for")
    parser.add_argument("--gamma_scheduler", type=float, help="factor by which to decay learning rate by")
    parser.add_argument("--lr", type=float,
                        help="initial learning rate to use during training (CURRENTLY NOT IMPLEMENTED)")
    parser.add_argument("--model_path", help="location to save/load a model state dict from")
    parser.add_argument("operation",
                        help="operation to perform: either TRAIN a new model or EVALUATE an existing model",
                        choices=["train", "evaluate"])
    parser.add_argument("--a", type=float, help="value of a")
    parser.add_argument("--b", type=float, help="value of b")
    parser.add_argument("--gamma", type=float, help="value of gamma")
    args = parser.parse_args()

    if args.operation == "train":

        deep_d.train_model(num_epochs=args.num_epochs, path=args.model_path, gamma_scheduler=args.gamma_scheduler,
                           batch_size=args.batch_size)

    else:
        evaluate.evaluate(args.model_path)
# scaler = joblib.load("scaler.save")
