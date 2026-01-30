# shell.nix
{ pkgs ? import <nixpkgs> {} }:
with pkgs; mkShell {
  name = "marimo_devel_env";

  buildInputs = [
    # Python mit benötigten Paketen
    glib
    libGL
    glibc
    zlib
    python313
    python313Packages.pandas
    python313Packages.tables
    python313Packages.h5py
    pkgs.fish # remove this if you still prefer bash
    # Zusätzliche Systemdependencies
    pkgs.binutils
    pkgs.stdenv.cc.cc.lib
    pkgs.uv
  ];

  # Umgebungsvariablen für Python und CUDA
  shellHook = ''



    # Create .venv if it doesn't exist
    if [ ! -d .venv ]; then
      echo "Creating .venv using uv's venv..."
      uv venv
    fi

    source .venv/bin/activate

    # If requirements.txt exists, install (only if packages not already installed)
    if [ -f requirements.txt ]; then
      echo "Installing Python requirements from requirements.txt into .venv (if needed)..."
      uv pip install -r requirements.txt
   fi

    # Friendly notice
    echo "Entered nix shell. Fish set as SHELL: $SHELL. Virtualenv at .venv is activated."

    
    # Standard environment
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib"

    # [CV2] Software / Hardware bridge
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${libGL}/lib"

    # [CV2] Useful data types
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.glib.out}/lib"

    # [Torch (CUDA)] OpenGL
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver-32/lib"

    # [Numpy] Data compression
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${pkgs.zlib.outPath}/lib"

    echo "Environment ready to run Notebooks"
    echo "start running marimo "
    
  '';

}
