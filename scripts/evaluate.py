""" Evaluation script for face verification model """

import argparse
import json
from src import face_verification

def main():
    parser = argparse.ArgumentParser(description="evaluation for face verification model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="path to the trainned model")
    
    parser.add_argument("--test-dir", type=str, required=True,
                        help="directory for the testing data")
    
    parser.add_argument("--identity-dir", type=str, required=True,
                        help="directory containing identity folders")
    
    parser.add_argument("--threshold", type=float, required=0.6,
                        help="similarity thershold")
    
    parser.add_argument("--output-file", type=str, required=True, 
                        help="output directory file results")
    
    parser.add_argument("--device", type=str, default="auto",
                        help="device to use (cuda), (mps) or (cpu)")
    
    args = parser.parse_args()

    # select the device
    device = "mps" if args.device == "auto" else args.device

    system = face_verification(device=device)

    # load the trainned model
    system.load_model(args.model_path)

    print(f"Evaluating model on test path")
    print(f"Test directory: {args.test_dir}")
    print(f"Identity directory: {args.identity_dir}")
    print(f"Threshold: {args.threshold}")

    # evaluate test set
    results = system.evaluate_test_set(
        test_dir = args.test_dir,
        identity_folder_dir=args.identity_dir,
        threshold=args.threshold
    )

    # calculate metrices
    total_images = len(results)
    matched_images = sum(1 for r in results if r["is_match"])
    accuracy = matched_images / total_images if total_images > 0 else 0

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump({
            'results': results,
            'summary': {
                'total_images': total_images,
                'matched_images': matched_images,
                'accuracy': accuracy,
                'threshold': args.threshold
            }
        }, f, indent=2)
    
    print(f"Evaluation completed!")
    print(f"Total images: {total_images}")
    print(f"Matched images: {matched_images}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Results saved to: {args.output_file}")


if __name__ == '__main__':
    main()

