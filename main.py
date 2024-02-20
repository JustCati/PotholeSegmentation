import os
import argparse
from createCOCO.createCOCO import convert




def main():
    parser = argparse.ArgumentParser(description="Pothole Segmentation")
    parser.add_argument("--path", type=str, default=os.path.join(os.getcwd(), "data"), help="Path to the data directory")
    parser.add_argument("--split", type=str, default="train", help="Target directory (train, test, val)", choices=["train", "test", "val"])
    parser.add_argument("--target", type=str, default="images", help="Target directory (images, videos)", choices=["images", "videos"])
    args = parser.parse_args()

    path = args.path
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    path = os.path.join(path, args.target, args.split)
    if not os.path.exists(path):
        raise ValueError(f"Path {path} does not exist")

    cocoPath = os.path.join(path, "cocoLabels.json")
    if not os.path.exists(cocoPath):
        convert(args)




if __name__ == "__main__":
    main()
