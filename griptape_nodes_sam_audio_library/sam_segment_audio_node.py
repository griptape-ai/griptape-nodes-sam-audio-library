import io
import logging

import torch
import torchaudio

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape.artifacts import AudioArtifact

logger = logging.getLogger("sam_audio_library")

MODEL_CHOICES = [
    "facebook/sam-audio-small",
    "facebook/sam-audio-base",
    "facebook/sam-audio-large",
]


class SamSegmentAudioNode(SuccessFailureNode):
    """Node for audio segmentation using SAM Audio."""

    _model = None
    _processor = None
    _current_model_id = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Model selection
        self.add_parameter(
            Parameter(
                name="model",
                allowed_modes={ParameterMode.PROPERTY},
                type="str",
                default_value="facebook/sam-audio-large",
                tooltip="SAM Audio model to use.",
                ui_options={
                    "options": MODEL_CHOICES,
                },
            )
        )

        # Input audio
        self.add_parameter(
            Parameter(
                name="audio",
                allowed_modes={ParameterMode.INPUT},
                type="AudioArtifact",
                default_value=None,
                tooltip="Input audio to segment.",
            )
        )

        # Text description for prompting
        self.add_parameter(
            Parameter(
                name="description",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="",
                tooltip="Text description of the sound to isolate (e.g., 'A man speaking', 'Piano playing').",
            )
        )

        # Anchors for span prompting
        self.add_parameter(
            Parameter(
                name="anchor_start",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=-1.0,
                tooltip="Start time in seconds for span prompting (-1 to disable).",
            )
        )

        self.add_parameter(
            Parameter(
                name="anchor_end",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=-1.0,
                tooltip="End time in seconds for span prompting (-1 to disable).",
            )
        )

        # Separation options
        self.add_parameter(
            Parameter(
                name="predict_spans",
                allowed_modes={ParameterMode.PROPERTY},
                type="bool",
                default_value=False,
                tooltip="Automatically predict time spans from text description. Improves results for non-ambient sounds.",
            )
        )

        self.add_parameter(
            Parameter(
                name="reranking_candidates",
                allowed_modes={ParameterMode.PROPERTY},
                type="int",
                default_value=1,
                tooltip="Number of candidates to generate and rerank. Higher values improve quality but increase latency (1-8).",
                ui_options={
                    "min": 1,
                    "max": 8,
                },
            )
        )

        # Output parameters
        self.add_parameter(
            Parameter(
                name="target_audio",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="AudioArtifact",
                default_value=None,
                tooltip="The isolated target sound.",
            )
        )

        self.add_parameter(
            Parameter(
                name="residual_audio",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="AudioArtifact",
                default_value=None,
                tooltip="The residual audio (everything except the target).",
            )
        )

        self.add_parameter(
            Parameter(
                name="sample_rate",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="int",
                default_value=48000,
                tooltip="Sample rate of the output audio.",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the audio segmentation result",
            result_details_placeholder="Segmentation result details will appear here.",
        )

    def _get_device(self) -> str:
        """Get the appropriate device for inference."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self, model_id: str) -> None:
        """Load the SAM Audio model and processor."""
        from sam_audio import SAMAudio, SAMAudioProcessor

        if SamSegmentAudioNode._model is not None and SamSegmentAudioNode._current_model_id == model_id:
            logger.info(f"Using cached model: {model_id}")
            return

        logger.info(f"Loading SAM Audio model: {model_id}")
        device = self._get_device()

        SamSegmentAudioNode._processor = SAMAudioProcessor.from_pretrained(model_id)
        SamSegmentAudioNode._model = SAMAudio.from_pretrained(model_id)
        SamSegmentAudioNode._model = SamSegmentAudioNode._model.eval().to(device)
        SamSegmentAudioNode._current_model_id = model_id

        logger.info(f"Model loaded on device: {device}")

    def _audio_artifact_to_tensor(self, artifact: AudioArtifact) -> torch.Tensor:
        """Convert an AudioArtifact to a torch tensor."""
        buffer = io.BytesIO(artifact.value)
        waveform, sample_rate = torchaudio.load(buffer)
        return waveform, sample_rate

    def _build_anchors(self) -> list | None:
        """Build anchors list from start/end parameters."""
        anchor_start = self.get_parameter_value("anchor_start")
        anchor_end = self.get_parameter_value("anchor_end")

        if anchor_start >= 0 and anchor_end >= 0 and anchor_end > anchor_start:
            return [[["+", anchor_start, anchor_end]]]

        return None

    def _tensor_to_audio_artifact(self, tensor: torch.Tensor, sample_rate: int) -> AudioArtifact:
        """Convert a torch tensor to an AudioArtifact."""
        # Ensure tensor is on CPU and has correct shape
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.cpu()

        # Save to bytes buffer
        buffer = io.BytesIO()
        torchaudio.save(buffer, tensor, sample_rate, format="wav")
        buffer.seek(0)

        return AudioArtifact(value=buffer.read(), format="wav")

    async def aprocess(self) -> None:
        await self._process()

    async def _process(self) -> None:
        self._clear_execution_status()

        model_id = self.get_parameter_value("model")
        audio_artifact = self.get_parameter_value("audio")
        description = self.get_parameter_value("description")
        predict_spans = self.get_parameter_value("predict_spans")
        reranking_candidates = self.get_parameter_value("reranking_candidates")

        if audio_artifact is None:
            self._set_status_results(
                was_successful=False,
                result_details="No audio input provided.",
            )
            return

        try:
            # Load model
            self.status_component.append_to_result_details(f"Loading model: {model_id}")
            self._load_model(model_id)

            model = SamSegmentAudioNode._model
            processor = SamSegmentAudioNode._processor
            device = self._get_device()

            # Convert artifact to tensor
            waveform, input_sample_rate = self._audio_artifact_to_tensor(audio_artifact)

            # Resample if needed (SAM Audio expects 48kHz)
            if input_sample_rate != processor.audio_sampling_rate:
                self.status_component.append_to_result_details(
                    f"Resampling from {input_sample_rate}Hz to {processor.audio_sampling_rate}Hz"
                )
                waveform = torchaudio.functional.resample(
                    waveform, input_sample_rate, processor.audio_sampling_rate
                )

            # Build processor inputs
            anchors = self._build_anchors()

            if anchors:
                self.status_component.append_to_result_details(
                    f"Using span prompting: {anchors[0][0][1]:.2f}s - {anchors[0][0][2]:.2f}s"
                )

            self.status_component.append_to_result_details(f"Description: {description or '(none)'}")

            # Process batch - pass tensor directly
            batch = processor(
                audios=[waveform],
                descriptions=[description or ""],
                anchors=anchors,
            ).to(device)

            # Run separation
            self.status_component.append_to_result_details(
                f"Running separation (predict_spans={predict_spans}, candidates={reranking_candidates})..."
            )

            with torch.inference_mode():
                result = model.separate(
                    batch,
                    predict_spans=predict_spans,
                    reranking_candidates=reranking_candidates,
                )

            sample_rate = processor.audio_sampling_rate

            # Convert outputs to artifacts
            target_tensor = result.target[0]
            residual_tensor = result.residual[0]

            target_artifact = self._tensor_to_audio_artifact(target_tensor, sample_rate)
            residual_artifact = self._tensor_to_audio_artifact(residual_tensor, sample_rate)

            # Set outputs
            self.set_parameter_value("target_audio", target_artifact)
            self.publish_update_to_parameter("target_audio", target_artifact)

            self.set_parameter_value("residual_audio", residual_artifact)
            self.publish_update_to_parameter("residual_audio", residual_artifact)

            self.set_parameter_value("sample_rate", sample_rate)
            self.publish_update_to_parameter("sample_rate", sample_rate)

            self._set_status_results(
                was_successful=True,
                result_details=f"Successfully separated audio. Sample rate: {sample_rate}Hz",
            )

        except Exception as e:
            msg = f"Error during audio segmentation: {e!s}"
            logger.exception(msg)
            self._set_status_results(was_successful=False, result_details=msg)
            return
