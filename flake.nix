{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }: 
  let
  	system = "aarch64-linux";
	pkgs = nixpkgs.legacyPackages.${system};
  in
  {
      devShells.${system}.default = pkgs.mkShell {
      	packages = with pkgs; [
		jdk21
		gradle
		perf
		linuxPackages.cpupower
		s6-linux-utils
	];
      };
  };
}
