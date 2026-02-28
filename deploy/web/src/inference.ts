/**
 * TeXer ONNX Runtime Web inference engine.
 *
 * Loads encoder and decoder ONNX models, preprocesses images,
 * and performs autoregressive decoding to produce LaTeX output.
 */

import * as ort from "onnxruntime-web";

export interface ModelInfo {
  image_size: number;
  max_seq_len: number;
  vocab_size: number;
  in_channels: number;
  encoder_file: string;
  decoder_file: string;
  vocab_file: string;
}

export interface Vocab {
  [token: string]: number;
}

export class TeXerEngine {
  private encoderSession: ort.InferenceSession | null = null;
  private decoderSession: ort.InferenceSession | null = null;
  private modelInfo: ModelInfo | null = null;
  private vocab: Vocab = {};
  private id2token: Map<number, string> = new Map();
  private bosId = 1;
  private eosId = 2;
  private padId = 0;

  async load(modelDir: string): Promise<void> {
    const infoResp = await fetch(`${modelDir}/model_info.json`);
    this.modelInfo = await infoResp.json();

    const vocabResp = await fetch(`${modelDir}/${this.modelInfo!.vocab_file}`);
    this.vocab = await vocabResp.json();

    for (const [token, id] of Object.entries(this.vocab)) {
      this.id2token.set(id as number, token);
    }
    this.bosId = this.vocab["<bos>"] ?? 1;
    this.eosId = this.vocab["<eos>"] ?? 2;
    this.padId = this.vocab["<pad>"] ?? 0;

    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    };

    this.encoderSession = await ort.InferenceSession.create(
      `${modelDir}/${this.modelInfo!.encoder_file}`,
      options
    );
    this.decoderSession = await ort.InferenceSession.create(
      `${modelDir}/${this.modelInfo!.decoder_file}`,
      options
    );
  }

  get isLoaded(): boolean {
    return this.encoderSession !== null && this.decoderSession !== null;
  }

  preprocessImage(imageData: ImageData): ort.Tensor {
    const size = this.modelInfo!.image_size;
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext("2d")!;

    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = imageData.width;
    tmpCanvas.height = imageData.height;
    const tmpCtx = tmpCanvas.getContext("2d")!;
    tmpCtx.putImageData(imageData, 0, 0);

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, size, size);

    const scale = Math.min(size / imageData.width, size / imageData.height);
    const w = imageData.width * scale;
    const h = imageData.height * scale;
    const x = (size - w) / 2;
    const y = (size - h) / 2;
    ctx.drawImage(tmpCanvas, x, y, w, h);

    const resized = ctx.getImageData(0, 0, size, size);
    const floatData = new Float32Array(size * size);

    for (let i = 0; i < size * size; i++) {
      const r = resized.data[i * 4];
      const g = resized.data[i * 4 + 1];
      const b = resized.data[i * 4 + 2];
      const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0;
      floatData[i] = (gray - 0.5) / 0.5;
    }

    return new ort.Tensor("float32", floatData, [1, 1, size, size]);
  }

  async recognize(imageData: ImageData, maxLen: number = 100): Promise<string> {
    if (!this.isLoaded) {
      throw new Error("Model not loaded. Call load() first.");
    }

    const imageTensor = this.preprocessImage(imageData);

    const encoderResult = await this.encoderSession!.run({
      image: imageTensor,
    });
    const encoderOutput = encoderResult["encoder_output"];

    let inputIds = [this.bosId];

    for (let step = 0; step < maxLen - 1; step++) {
      const inputTensor = new ort.Tensor(
        "int64",
        BigInt64Array.from(inputIds.map((id) => BigInt(id))),
        [1, inputIds.length]
      );

      const decoderResult = await this.decoderSession!.run({
        input_ids: inputTensor,
        encoder_output: encoderOutput,
      });

      const logits = decoderResult["logits"];
      const logitsData = logits.data as Float32Array;
      const vocabSize = this.modelInfo!.vocab_size;
      const lastStepOffset = (inputIds.length - 1) * vocabSize;

      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let i = 0; i < vocabSize; i++) {
        const val = logitsData[lastStepOffset + i];
        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }

      if (maxIdx === this.eosId || maxIdx === this.padId) break;
      inputIds.push(maxIdx);
    }

    return this.decodeTokens(inputIds);
  }

  private decodeTokens(ids: number[]): string {
    const tokens: string[] = [];
    for (const id of ids) {
      if (id === this.bosId || id === this.eosId || id === this.padId) continue;
      const token = this.id2token.get(id) ?? "";
      if (token) tokens.push(token);
    }
    return tokens.join(" ");
  }
}
