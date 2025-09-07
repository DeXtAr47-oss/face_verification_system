""" Inference script for face verification model """
import argparse
from src import face_verification

def main():
    parser = argparse.ArgumentParser(description="face verification inference")
    parser.add_argument('--model-path', type=str, required=True,
                        help="path to trainned model")
    
    parser.add_argument('--test-image', type=str, required=True,
                        help="path to the test image")
    
    parser.add_argument('--identity-folder', type=str, required=True,
                        help="path to the identtiy folder")
    
    parser.add_argument('--threshold', type=float, default=0.6, 
                        help="similarity threshold")
    
    parser.add_argument('--device', type=str, default='auto',
                        help='device to use (mps), (cuda) or (cpu)')
    
    args = parser.parse_args()

    # initilize system
    device = 'mps' if args.device == 'auto' else args.device
    system =  face_verification(device = device)

    # load trainned model
    system.load_model(args.model_path)

    print(f"verifying identtiy..")
    print(f"Test image path: {args.test_img}")
    print(f"Identity folder: {args.identity_folder}")
    print(f"threshold: {args.threshold}")

    # perform validation
    is_match, confidence, distance = system.verify_identity(
        test_iamge=args.test_image,
        identity_folder=args.identity_folder,
        threshold=args.threshold
    )

    print("\n-----Verification Result------")
    print(f"Match: {'YES' if is_match else "NO"}")
    print(f"Confidence: {confidence: .4f}")
    print(f"Threshold: {args.threshold}")

if __name__ == "__main__":
    main()
    