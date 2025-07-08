
pip install -e ".[eval]"
# Override transformers and vllm versions, bypassing pyproject conflicts with olmes.
# TODO: remove once merged
pip uninstall vllm; pip uninstall -y transformers; pip install "transformers@git+https://github.com/swj0419/transformers"; pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124; pip install ipdb;