# SAM Audio Nodes for Griptape

This library provides Griptape Nodes for [SAM Audio](https://github.com/facebookresearch/sam-audio), Meta's Segment Anything Model for audio. It lets you isolate specific sounds from an audio track using text, temporal, or span prompts directly within your Griptape workflows.

The models run locally on your own hardware and require a CUDA-capable GPU.

## Features

- **Prompted sound isolation**: Extract a target sound from a mixed audio track using a natural-language description (e.g. `"A man speaking"`, `"Piano playing"`).
- **Span / anchor prompting**: Specify time ranges where the target sound occurs, and choose whether to include or exclude those spans.
- **Automatic span prediction**: Let the model predict the relevant time spans from your text description, improving results for non-ambient sounds.
- **Candidate reranking**: Generate and rerank multiple candidates to improve output quality.
- **Dual outputs**: Get both the isolated target sound and the residual audio (everything except the target).
- **Model selection**: Choose between `sam-audio-small`, `sam-audio-base`, and `sam-audio-large` depending on your quality and VRAM needs.

## Requirements

- A CUDA-capable NVIDIA GPU.
- The SAM Audio model weights are downloaded automatically from Hugging Face on first use.

## Installation

1. Clone this repository into your Griptape Nodes workspace directory:

```bash
# Navigate to your workspace directory
# On Mac or Linux you can use the command below to print your workspace directory
cd $(gtn config show | grep workspace_directory | cut -d'"' -f4)
# On Windows, the default workspace directory is a directory named GriptapeNodes in your home directory.
# Usually this is C:\Users\<username>\GriptapeNodes

# Clone the repository
git clone https://github.com/griptape-ai/griptape-nodes-sam-audio-library.git
```

2. Install dependencies:

```bash
cd griptape-nodes-sam-audio-library
uv sync
```

## Add your library to your installed Engine!

If you haven't already installed your Griptape Nodes engine, follow the installation steps [HERE](https://github.com/griptape-ai/griptape-nodes).
After you've completed those and you have your engine up and running:

1. Copy the path to your `griptape-nodes-library.json` file within the `griptape_nodes_sam_audio_library` directory. Right click on the file, and `Copy Path` (Not `Copy Relative Path`).
2. Start up the engine!
3. Navigate to settings.
4. Open your settings and go to the App Events tab. Add an item in **Libraries to Register**.
5. Paste your copied `griptape-nodes-library.json` path from earlier into the new item.
6. Exit out of Settings. It will save automatically!
7. Open up the **Libraries** dropdown on the left sidebar.
8. Your newly registered library should appear! Drag and drop nodes to use them!

## Available Nodes

### SAM Segment Audio

Isolate a specific sound from an audio track using text, temporal, or span prompts.

- **Model**: Choose from:
  - `facebook/sam-audio-small`
  - `facebook/sam-audio-base`
  - `facebook/sam-audio-large`
- **Audio**: The input audio to segment (accepts `AudioArtifact` and `AudioUrlArtifact`).
- **Description**: Text description of the sound to isolate (e.g. `"A man speaking"`, `"Piano playing"`).
- **Use Anchors**: Enable span/anchor prompting to specify time ranges where the target sound occurs.
- **Anchor Token**: `+` to include the sound in the span, `-` to exclude it.
- **Anchor Start / Anchor End**: Start and end time in seconds for span prompting.
- **Predict Spans**: Automatically predict time spans from the text description. Improves results for non-ambient sounds.
- **Reranking Candidates**: Number of candidates to generate and rerank (1-8). Higher values improve quality but increase latency.

Outputs:

- **Target Audio**: The isolated target sound.
- **Residual Audio**: The residual audio (everything except the target).
- **Sample Rate**: Sample rate of the output audio.

## Example Workflow

### Isolate a Voice from a Mixed Track

1. Add a **SAM Segment Audio** node.
2. Connect your source audio to the **Audio** input.
3. Set the **Description** to describe the sound you want, e.g. `"A man speaking"`.
4. Run the workflow.
5. The isolated voice is available on the **Target Audio** output, and the remaining background on the **Residual Audio** output.

### Target a Specific Time Range

1. Enable **Use Anchors**.
2. Set **Anchor Start** and **Anchor End** to the time range (in seconds) where your target sound occurs.
3. Set **Anchor Token** to `+` to isolate the sound within that span, or `-` to exclude it.
4. Run the workflow.

## Troubleshooting

**"CUDA not available"**
- SAM Audio requires a CUDA-capable NVIDIA GPU. Confirm your drivers and CUDA runtime are installed.

**Slow first run**
- The model weights are downloaded from Hugging Face on first use and cached locally. Subsequent runs are faster.

**Poor isolation quality**
- Try a larger model (`sam-audio-large`), enable **Predict Spans**, or increase **Reranking Candidates**.

## License

The Griptape Nodes integration code in this repository is provided by Griptape, Inc. The SAM Audio model and materials are distributed by Meta under the SAM License - see the [LICENSE](LICENSE) file for details. By using this library you agree to the terms of that license.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
