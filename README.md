# facial-recognition
### A facial detection and recognition application built using concepts from FaceNet and the Viola-Jones object detection framework

### This project consists of two parts:
  1. Facial detection using the Viola-Jones object detection framework implemented in rust from scratch.
  2. Facial recognintion using the official tensorflow binding for python with a PyQt5 frontend.

### Training the facial-detection program:
  1. Install Rust/Cargo or enable direnv if your system contains Nix/NixOS with flakes enabled
  2. Optionally change the widow size used in training and detection by editing the constant WS defined in src/primitives.rs (larger values of WS means more features which means more memory use and a longer training time)
  4. Run `cargo run --release -- cascade`
  5. After some time,  
### Using the facial-detection program:
  1. Install Rust/Cargo or enable direnv if your system contains Nix/NixOS with flakes enabled
  2. Run `cargo run --release -- detect /path/to/img.png` in the root directoy of the repository where /path/to/img.png can be any path to an image of any name with any common format (png, jpeg, etc.)
  3. A copy of the input image will be outputted with a red rectangle around will in the output directory with the same name as the original.

### Building and training the facial-recognition neural network (unimplemented):
  1. Install python3 and pip
  2. Run `pip install tensorflow` (Any python virtual environment with the specified packages can also be used)
  3. Run `python src/train.py`

### Using the facial-recognition neural network (unimplemented):
  1. Install python3 and pip
  2. Run `pip install tensorflow PyQt5` (Any python virtual environment with the specified packages can also be used)
  3. Run `python src/main.py`
