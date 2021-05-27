# object-detection
### An object detection application built using concepts from the Viola-Jones object detection algorithm

### Training the object detection program:
  1. Install Rust/Cargo or enable direnv if your system contains Nix/NixOS with flakes enabled
  2. Customize training and detection by editing the constants in src/constants.rs
  3. Run `cargo run --release -- process_images`
  4. Run `cargo run --release -- cascade` (This will take a long time)
  5. The cascade will be serialized to json and outputted in the location specified in src/constants.rs 

### Using the object detection program:
  1. Install Rust/Cargo or enable direnv if your system contains Nix/NixOS with flakes enabled
  2. Run `cargo run --release -- detect /path/to/img.png` in the root directoy of the repository where /path/to/img.png can be any path to an image of any name with any common format (png, jpeg, etc.)
  3. A copy of the input image will be outputted with a red rectangle around found instances of the object will be created in the output directory specified in src/constants.rs with the same name as the original.


## Important information:
  - The positive traininig images must be of the same aspect ratio. They should also be cropped to the object. Additionally, you must edit the aspect ratio between WL and WH in src/constants.rs to match that of the positive training images.
  - You should not change the parameters after building the cascade especially the ones related to the window size.
  - Choose MAX_FALSE_NEG based on CASCADE_SIZE and the desired false negative rate for the cascade as detailed in src/constants.rs. Higher-values typically mean longer training times, but faster detection times.
  - It is typically better to have more negative training samples than positive training samples.
