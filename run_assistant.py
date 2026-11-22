"""
Executable Entry Point: Sign Language Recognition and Translation Assistant
Launches the interactive real-time assistant application with camera capture and speech synthesis.
"""

import argparse
import os
import yaml
import numpy as np
import torch

from src.assistant.real_time_app import SignLanguageAssistant
from src.data.synthetic_generator import generate_gesture_trajectory
from src.models.bilstm_attention import BiLSTMTemporalAttention
from src.training.utils import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Sign Language Recognition & Translation Assistant")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/bilstm_attention_best.pt", help="Model checkpoint path")
    parser.add_argument("--no-tts", action="store_true", help="Disable audio speech output")
    parser.add_argument("--test-mode", action="store_true", help="Run simulated test mode instead of live webcam")
    return parser.parse_args()


def run_simulated_stream(assistant, classes):
    """Simulates a live webcam feed using synthetic gesture sequences for testing."""
    import cv2

    print("--- Running Simulated Live Stream (Test Mode: 90 frames) ---")
    # Simulate a sequence: 30 frames of 'Hello' (class 0), 30 frames of 'Help' (class 4), 30 frames of 'Thank You' (class 1)
    simulated_classes = [0, 4, 1]
    for c in simulated_classes:
        print(f"\n[Simulator]: Injecting gesture '{classes[c]}'...")
        traj = generate_gesture_trajectory(class_idx=c, num_frames=30)
        for frame_idx in range(30):
            # Create a mock video frame (black background with test label)
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"Simulated Signer: {classes[c]} (Frame {frame_idx+1}/30)",
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            # Inject precomputed features into assistant buffer
            assistant.buffer.append(traj[frame_idx])
            # Process frame for HUD rendering
            hud_frame = assistant.process_frame(dummy_frame)

    print(f"\n[Simulator Completed]: Final sentence transcript: \"{assistant.debouncer.get_sentence()}\"")


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    classes = config["data"]["classes"]
    feature_dim = config["data"]["feature_dim"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading assistant model from {args.checkpoint}...")

    model = BiLSTMTemporalAttention(
        input_dim=feature_dim,
        spatial_embed_dim=config["models"]["bilstm_attention"]["spatial_embed_dim"],
        hidden_dim=config["models"]["bilstm_attention"]["hidden_dim"],
        num_layers=config["models"]["bilstm_attention"]["num_layers"],
        num_classes=len(classes),
        recurrent_dropout=config["models"]["bilstm_attention"]["recurrent_dropout"],
        dense_dim=config["models"]["bilstm_attention"]["dense_dim"],
        fc_dropout=config["models"]["bilstm_attention"]["fc_dropout"],
    )

    if os.path.exists(args.checkpoint):
        load_checkpoint(args.checkpoint, model)
        print("Model checkpoint loaded successfully.")
    else:
        print(f"[Warning]: Checkpoint '{args.checkpoint}' not found. Using initialized weights for demo.")

    assistant = SignLanguageAssistant(
        model=model,
        class_names=classes,
        confidence_threshold=config["assistant"]["confidence_threshold"],
        debounce_frames=config["assistant"]["debounce_frames"],
        camera_index=args.camera,
        enable_tts=not args.no_tts,
        device=device,
    )

    if args.test_mode:
        run_simulated_stream(assistant, classes)
    else:
        assistant.run()


if __name__ == "__main__":
    main()
