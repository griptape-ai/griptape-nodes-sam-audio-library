import logging
import subprocess
import sys
from pathlib import Path

import pygit2

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam_audio_library")


class SamAudioLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for SAM Audio."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        logger.info("Initializing sam-audio submodule...")
        sam_audio_path = self._init_sam_audio_submodule()

        logger.info("Installing sam-audio dependencies...")
        self._install_sam_audio_dependencies(sam_audio_path)

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

    def _install_sam_audio_dependencies(self, sam_audio_path: Path) -> None:
        """Install SAM Audio and its dependencies from the submodule."""
        try:
            # Check if sam_audio is already importable
            import sam_audio
            logger.info("sam-audio already installed")
            return
        except ImportError:
            pass

        logger.info(f"Installing sam-audio from {sam_audio_path}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", str(sam_audio_path)
        ])
        logger.info("sam-audio installed successfully")
