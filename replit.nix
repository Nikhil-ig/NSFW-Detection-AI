{ pkgs }: 

pkgs.mkShell rec {
  buildInputs = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.setuptools
    pkgs.python310Packages.wheel
  ];

  shellHook = ''
    export PYTHONPATH=$PYTHONPATH:${pkgs.python310Packages.pillow}/lib/python3.10/site-packages
    export PATH=$PATH:${pkgs.python310Packages.pillow}/bin
  '';
}
