import logging

from griptape_nodes.node_library.advanced_node_library import AdvancedNodeLibrary
from griptape_nodes.node_library.library_registry import Library, LibrarySchema

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam_audio_library")


def _patch_sam_audio_for_new_huggingface_hub() -> None:
    """Patch sam_audio BaseModel to be compatible with huggingface-hub >= 1.0."""
    try:
        from sam_audio.model.base import BaseModel
    except ImportError:
        # Expected on the orchestrator, where sam_audio lives in the execution environment that
        # only a worker receives. The patch is needed where from_pretrained runs, which is there.
        logger.debug("sam_audio not importable in this process, skipping huggingface-hub compatibility patch")
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


class SamAudioLibraryAdvanced(AdvancedNodeLibrary):
    """Advanced library implementation for SAM Audio."""

    def before_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called before any nodes are loaded from the library."""
        msg = f"Starting to load nodes for '{library_data.name}' library..."
        logger.info(msg)

        # Patch sam_audio for compatibility with huggingface-hub >= 1.0
        _patch_sam_audio_for_new_huggingface_hub()

    def after_library_nodes_loaded(self, library_data: LibrarySchema, library: Library) -> None:
        """Called after all nodes have been loaded from the library."""
        msg = f"Finished loading nodes for '{library_data.name}' library"
        logger.info(msg)
