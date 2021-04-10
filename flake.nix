{
  outputs = { self, nixpkgs }:
      let pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
      };
  in {
    devShell.x86_64-linux = pkgs.mkShell {
      buildInputs = with pkgs; with python38Packages;
      [
        # Rust Dependencies
        cargo rustfmt clippy 

        # Python Dependencies
        python38 pipenv setuptools stdenv.cc.cc.lib
      ];
      shellHook = ''
		export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.cudatoolkit_11_2}/lib:${pkgs.cudnn_cudatoolkit_11_2}/lib:$LD_LIBRARY_PATH
      '';
    };
  };
}
