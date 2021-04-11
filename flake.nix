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
        python38 scrapy
      ];
    };
  };
}
