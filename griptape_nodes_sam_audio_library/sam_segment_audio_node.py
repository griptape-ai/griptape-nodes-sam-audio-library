import io
import logging

import torch
import torchaudio
from griptape.artifacts import AudioArtifact, AudioUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.traits.options import Options

logger = logging.getLogger("sam_audio_library")

MODEL_REPO_IDS = [
    "facebook/sam-audio-small",
    "facebook/sam-audio-base",
    "facebook/sam-audio-large",
]

ANCHOR_TOKEN_CHOICES = [
    "+",
    "-",
]


class SamSegmentAudioNode(SuccessFailureNode):
    """Node for audio segmentation using SAM Audio."""

    _model = None
    _processor = None
    _current_model_id = None

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Model selection using HuggingFace repo parameter
        self._model_repo_parameter = HuggingFaceRepoParameter(
            self,
            repo_ids=MODEL_REPO_IDS,
            parameter_name="model",
        )
        self._model_repo_parameter.add_input_parameters()

        # Input audio
        self.add_parameter(
            Parameter(
                name="audio",
                allowed_modes={ParameterMode.INPUT},
                type="AudioArtifact",
                input_types=["AudioArtifact", "AudioUrlArtifact"],
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
                name="use_anchors",
                allowed_modes={ParameterMode.PROPERTY},
                type="bool",
                default_value=False,
                tooltip="Enable span/anchor prompting to specify time ranges where the target sound occurs.",
            )
        )

        self.add_parameter(
            Parameter(
                name="anchor_token",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="str",
                default_value="+",
                tooltip="Anchor token: '+' to include the sound in the span, '-' to exclude it.",
                traits={Options(choices=ANCHOR_TOKEN_CHOICES)},
                ui_options={
                    "hide": True,
                },
            )
        )

        self.add_parameter(
            Parameter(
                name="anchor_start",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=0.0,
                tooltip="Start time in seconds for span prompting.",
                ui_options={
                    "hide": True,
                },
            )
        )

        self.add_parameter(
            Parameter(
                name="anchor_end",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="float",
                default_value=1.0,
                tooltip="End time in seconds for span prompting.",
                ui_options={
                    "hide": True,
                },
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
                    "slider": {"min_val": 1, "max_val": 8, "step": 1},
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

    def _set_parameter_visibility(self, names: str | list[str], *, visible: bool) -> None:
        """Sets the visibility of one or more parameters.

        Args:
            names: The parameter name(s) to update.
            visible: Whether to show (True) or hide (False) the parameters.
        """
        if isinstance(names, str):
            names = [names]

        for name in names:
            parameter = self.get_parameter_by_name(name)
            if parameter is not None:
                ui_options = parameter.ui_options
                ui_options["hide"] = not visible
                parameter.ui_options = ui_options

    def after_value_set(self, parameter: Parameter, value) -> None:
        """Handle parameter value changes to update anchor/predict_spans visibility."""
        if parameter.name == "use_anchors":
            anchor_params = ["anchor_token", "anchor_start", "anchor_end"]
            self._set_parameter_visibility(anchor_params, visible=value)
            # Hide predict_spans when using manual anchors (they serve similar purposes)
            self._set_parameter_visibility("predict_spans", visible=not value)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that the HuggingFace model is available."""
        return self._model_repo_parameter.validate_before_node_run()

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

    def _audio_artifact_to_tensor(self, artifact: AudioArtifact | AudioUrlArtifact) -> tuple[torch.Tensor, int]:
        """Convert an AudioArtifact or AudioUrlArtifact to a torch tensor."""
        if isinstance(artifact, AudioUrlArtifact):
            # Download URL content to bytes, then load from buffer
            audio_bytes = artifact.to_bytes()
            buffer = io.BytesIO(audio_bytes)
        else:
            # Load from bytes directly
            buffer = io.BytesIO(artifact.value)
        waveform, sample_rate = torchaudio.load(buffer)
        return waveform, sample_rate

    def _build_anchors(self) -> list | None:
        """Build anchors list from parameters."""
        use_anchors = self.get_parameter_value("use_anchors")
        if not use_anchors:
            return None

        anchor_token = self.get_parameter_value("anchor_token")
        anchor_start = self.get_parameter_value("anchor_start")
        anchor_end = self.get_parameter_value("anchor_end")

        if anchor_end > anchor_start:
            return [[[anchor_token, anchor_start, anchor_end]]]

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
                waveform = torchaudio.functional.resample(waveform, input_sample_rate, processor.audio_sampling_rate)

            # Build processor inputs
            anchors = self._build_anchors()

            if anchors:
                token, start, end = anchors[0][0]
                self.status_component.append_to_result_details(
                    f"Using span prompting: token='{token}', {start:.2f}s - {end:.2f}s"
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
            self.parameter_output_values["target_audio"] = target_artifact
            self.parameter_output_values["residual_audio"] = residual_artifact
            self.parameter_output_values["sample_rate"] = sample_rate

            self._set_status_results(
                was_successful=True,
                result_details=f"Successfully separated audio. Sample rate: {sample_rate}Hz",
            )

        except Exception as e:
            msg = f"Error during audio segmentation: {e!s}"
            logger.exception(msg)
            self._set_status_results(was_successful=False, result_details=msg)
            return
