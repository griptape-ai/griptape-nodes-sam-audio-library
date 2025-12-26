import logging
import subprocess
import sys
from pathlib import Path

import pygit2

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam_audio_library")


def _get_library_venv_python() -> Path:
    """Get the path to the library venv's Python executable."""
    library_root = Path(__file__).parent
    if sys.platform == "win32":
        venv_python = library_root / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = library_root / ".venv" / "bin" / "python"
    return venv_python


def _get_library_site_packages() -> Path:
    """Get the path to the library venv's site-packages."""
    library_root = Path(__file__).parent
    if sys.platform == "win32":
        return library_root / ".venv" / "Lib" / "site-packages"
    else:
        # Find the python version directory
        lib_dir = library_root / ".venv" / "lib"
        if lib_dir.exists():
            for p in lib_dir.iterdir():
                if p.name.startswith("python"):
                    return p / "site-packages"
        return lib_dir / "python3" / "site-packages"


def _ensure_pip_installed() -> None:
    """Ensure pip is installed in the library venv."""
    venv_python = _get_library_venv_python()
    if not venv_python.exists():
        logger.warning(f"Library venv not found at {venv_python}, skipping pip check")
        return

    # Check if pip is available
    result = subprocess.run(
        [str(venv_python), "-m", "pip", "--version"],
        capture_output=True
    )
    if result.returncode == 0:
        logger.info("pip is available in library venv")
        return

    logger.info("pip not found in library venv, installing with ensurepip...")
    subprocess.check_call([
        str(venv_python), "-m", "ensurepip", "--upgrade"
    ])
    logger.info("pip installed successfully")


def _install_perception_models_no_deps() -> None:
    """Install perception-models with --no-deps to skip torchcodec dependency on Windows."""
    if sys.platform != "win32":
        return

    venv_python = _get_library_venv_python()
    if not venv_python.exists():
        logger.warning(f"Library venv not found at {venv_python}, skipping perception-models install")
        return

    # Ensure pip is available first
    _ensure_pip_installed()

    # Check if already installed
    result = subprocess.run(
        [str(venv_python), "-c", "import core"],
        capture_output=True
    )
    if result.returncode == 0:
        logger.info("perception-models already installed")
        return

    logger.info("Installing perception-models with --no-deps (Windows workaround)...")
    subprocess.check_call([
        str(venv_python), "-m", "pip", "install",
        "--no-deps",
        "perception-models@git+https://github.com/facebookresearch/perception_models@unpin-deps"
    ])
    logger.info("perception-models installed successfully")


def _setup_torchcodec_mock() -> None:
    """Create mock torchcodec package files for Windows (torchcodec has no Windows wheels)."""
    if sys.platform != "win32":
        return

    # Use the library venv's site-packages
    site_packages = _get_library_site_packages()
    if not site_packages.exists():
        logger.warning(f"Library site-packages not found at {site_packages}, skipping torchcodec mock")
        return

    torchcodec_dir = site_packages / "torchcodec"

    if torchcodec_dir.exists():
        logger.info("torchcodec mock already exists")
        return

    logger.info("Creating torchcodec mock package for Windows...")

    # Create package directories
    torchcodec_dir.mkdir(exist_ok=True)
    (torchcodec_dir / "decoders").mkdir(exist_ok=True)
    (torchcodec_dir / "encoders").mkdir(exist_ok=True)

    # Main __init__.py
    (torchcodec_dir / "__init__.py").write_text('__version__ = "0.0.0.dev0"\n')

    # decoders/__init__.py
    (torchcodec_dir / "decoders" / "__init__.py").write_text('''
class VideoDecoder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VideoDecoder requires torchcodec which is not available on Windows. "
            "Use text prompting instead of visual/video prompting."
        )

class AudioDecoder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AudioDecoder from torchcodec is not available on Windows. "
            "Audio loading via torchaudio should still work for most use cases."
        )
''')

    # encoders/__init__.py
    (torchcodec_dir / "encoders" / "__init__.py").write_text('''
class AudioEncoder:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "AudioEncoder requires torchcodec which is not available on Windows."
        )
''')

    # Create dist-info for package metadata
    dist_info = site_packages / "torchcodec-0.0.0.dev0.dist-info"
    dist_info.mkdir(exist_ok=True)
    (dist_info / "METADATA").write_text("""Metadata-Version: 2.1
Name: torchcodec
Version: 0.0.0.dev0
Summary: Mock torchcodec for Windows
""")
    (dist_info / "INSTALLER").write_text("pip\n")
    (dist_info / "RECORD").write_text("")

    logger.info("torchcodec mock package created successfully")


def _patch_sam_audio_for_new_huggingface_hub() -> None:
    """Patch sam_audio BaseModel to be compatible with huggingface-hub >= 1.0."""
    try:
        from sam_audio.model.base import BaseModel
    except ImportError:
        logger.warning("sam_audio not yet importable, skipping huggingface-hub compatibility patch")
        return

    original_from_pretrained = BaseModel._from_pretrained

    @classmethod
    def patched_from_pretrained(
        cls,
        *,
        model_id: str,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=None,
        local_files_only=False,
        token=None,
        map_location: str = "cpu",
        strict: bool = True,
        revision=None,
        **model_kwargs,
    ):
        return original_from_pretrained.__func__(
            cls,
            model_id=model_id,
            cache_dir=cache_dir or None,
            force_download=force_download or False,
            proxies=proxies or None,
            resume_download=resume_download if resume_download is not None else False,
            local_files_only=local_files_only or False,
            token=token or None,
            map_location=map_location or "cpu",
            strict=strict if strict is not None else True,
            revision=revision or None,
            **model_kwargs,
        )

    BaseModel._from_pretrained = patched_from_pretrained
    logger.info("Patched sam_audio BaseModel for huggingface-hub >= 1.0 compatibility")


def _clear_cached_modules() -> None:
    """Clear cached modules so they get re-imported from the library venv.

    This is necessary because huggingface_hub and transformers may be imported
    by the main griptape-nodes engine BEFORE the library venv is added to sys.path.
    Once cached in sys.modules, Python reuses them regardless of path order.
    """
    modules_to_clear = []

    # Find all huggingface_hub and transformers modules to clear
    for module_name in list(sys.modules.keys()):
        if module_name == "huggingface_hub" or module_name.startswith("huggingface_hub."):
            modules_to_clear.append(module_name)
        elif module_name == "transformers" or module_name.startswith("transformers."):
            modules_to_clear.append(module_name)

    if modules_to_clear:
        logger.info(f"Clearing {len(modules_to_clear)} cached modules to use library venv versions")
        for module_name in modules_to_clear:
            del sys.modules[module_name]
        logger.info("Cleared modules: huggingface_hub.*, transformers.*")


class SamAudioLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for SAM Audio."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        # Clear cached modules so we use the library venv versions
        # This MUST happen before any imports of huggingface_hub or transformers
        _clear_cached_modules()

        # Set up torchcodec mock for Windows before any sam_audio imports
        _setup_torchcodec_mock()

        # Install perception-models with --no-deps on Windows (to skip torchcodec)
        _install_perception_models_no_deps()

        logger.info("Initializing sam-audio submodule...")
        sam_audio_path = self._init_sam_audio_submodule()

        # Install sam_audio package from submodule (--no-deps since JSON handles dependencies)
        self._install_sam_audio(sam_audio_path)

        # Patch sam_audio for compatibility with huggingface-hub >= 1.0
        _patch_sam_audio_for_new_huggingface_hub()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)

    def _get_library_root(self) -> Path:
        """Get the library root directory."""
        return Path(__file__).parent

    def _update_submodules_recursive(self, repo_path: Path) -> None:
        """Recursively update and initialize all submodules."""
        repo = pygit2.Repository(str(repo_path))
        repo.submodules.update(init=True)

        for submodule in repo.submodules:
            submodule_path = repo_path / submodule.path
            if submodule_path.exists() and (submodule_path / ".git").exists():
                self._update_submodules_recursive(submodule_path)

    def _init_sam_audio_submodule(self) -> Path:
        """Initialize the sam-audio git submodule."""
        library_root = self._get_library_root()
        sam_audio_submodule_dir = library_root / "sam-audio"

        if sam_audio_submodule_dir.exists() and any(sam_audio_submodule_dir.iterdir()):
            logger.info("sam-audio submodule already initialized")
            return sam_audio_submodule_dir

        git_repo_root = library_root.parent
        self._update_submodules_recursive(git_repo_root)

        if not sam_audio_submodule_dir.exists() or not any(sam_audio_submodule_dir.iterdir()):
            raise RuntimeError(
                f"Submodule initialization failed: {sam_audio_submodule_dir} is empty or does not exist"
            )

        logger.info("sam-audio submodule initialized successfully")
        return sam_audio_submodule_dir

    def _get_venv_python_path(self) -> Path:
        """Get the path to the library venv's Python executable."""
        library_root = self._get_library_root()
        if sys.platform == "win32":
            venv_python = library_root / ".venv" / "Scripts" / "python.exe"
        else:
            venv_python = library_root / ".venv" / "bin" / "python"

        if not venv_python.exists():
            raise RuntimeError(f"Library venv Python not found at {venv_python}")
        return venv_python

    def _ensure_pip_installed(self) -> None:
        """Ensure pip is installed in the library venv."""
        venv_python = self._get_venv_python_path()

        # Check if pip is available
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "--version"],
            capture_output=True
        )
        if result.returncode == 0:
            logger.info("pip is available in library venv")
            return

        logger.info("pip not found in library venv, installing with ensurepip...")
        subprocess.check_call([
            str(venv_python), "-m", "ensurepip", "--upgrade"
        ])
        logger.info("pip installed successfully")

    def _install_sam_audio(self, sam_audio_path: Path) -> None:
        """Install sam_audio from the submodule into the library venv."""
        venv_python = self._get_venv_python_path()

        # Ensure pip is available first
        self._ensure_pip_installed()

        # Check if already installed in library venv
        result = subprocess.run(
            [str(venv_python), "-c", "import sam_audio"],
            capture_output=True
        )
        if result.returncode == 0:
            logger.info("sam_audio already installed in library venv")
            return

        logger.info(f"Installing sam_audio from {sam_audio_path}...")
        subprocess.check_call([
            str(venv_python), "-m", "pip", "install",
            "--no-deps",
            str(sam_audio_path)
        ])
        logger.info("sam_audio installed successfully")
