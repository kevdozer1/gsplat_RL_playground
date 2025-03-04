@echo off
echo Setting up PLAYGROUND environment...

:: Activate virtual environment if it exists, or create it(IMPORTANT: THIS IS NOT FOR WSL. WSL USES wsl_env and has this stuff already installed. It did run into a little problem with pybullet though)
if exist playground_env\Scripts\activate.bat (
    echo Activating existing virtual environment...
    call playground_env\Scripts\activate.bat
) else ( 
    echo Creating new virtual environment...
    python -m venv playground_env
    call playground_env\Scripts\activate.bat
)

:: Upgrade pip
python -m pip install --upgrade pip

:: Install required packages with CUDA support for PyTorch
echo Installing required packages...
pip install pybullet gymnasium stable-baselines3 numpy matplotlib trimesh PyYAML tensorboard

:: Install PyTorch with CUDA support
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Setup complete!
echo.
echo You can now run:
echo python standalone_train.py             - To train a model
echo python run_model.py --model MODEL_PATH - To run a trained model
echo.