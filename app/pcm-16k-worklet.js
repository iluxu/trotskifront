// pcm-16k-worklet.js
class PCM16kWriter extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opt = (options && options.processorOptions) || {};
    this.inRate = sampleRate;                 // ex: 48000
    this.outRate = opt.targetSampleRate || 16000;
    this.chunkMs = opt.chunkMs || 100;        // ~0.1 s
    this.channels = 1;                        // mono pour Whisper
    this._buf = [];                           // Float32 accumulé (mono)
    this._outSampPerChunk = Math.round(this.outRate * (this.chunkMs / 1000));

    this.port.postMessage({ type: 'ready', inRate: this.inRate, outRate: this.outRate });
  }

  // Downsample naïf (suffit pour STT)
  downsampleFloat32_(input, inRate, outRate) {
    if (outRate === inRate) return input;
    const ratio = inRate / outRate;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    let pos = 0;
    for (let i = 0; i < outLen; i++) {
      out[i] = input[Math.floor(pos)];
      pos += ratio;
    }
    return out;
  }

  toInt16LE_(floats) {
    const out = new Int16Array(floats.length);
    for (let i = 0; i < floats.length; i++) {
      let s = Math.max(-1, Math.min(1, floats[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return out;
  }

  process(inputs) {
    // inputs[0] = tableau des canaux
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    // Mono : si stéréo, on moyenne L/R
    const ch0 = input[0] || new Float32Array();
    const ch1 = input[1];
    let mono = ch1 ? this.mixToMono_(ch0, ch1) : ch0;

    // downsample → 16k
    const ds = this.downsampleFloat32_(mono, this.inRate, this.outRate);

    // accumulate
    this._buf.push(ds);
    const total = this.totalLength_(this._buf);
    if (total >= this._outSampPerChunk) {
      // concat jusqu'à chunk
      const take = this._outSampPerChunk;
      const chunkF32 = new Float32Array(take);
      this.drain_(this._buf, chunkF32);
      const pcm16 = this.toInt16LE_(chunkF32);

      // transfère le buffer vers le thread principal pour envoi WS
      this.port.postMessage({ type: 'chunk', payload: pcm16.buffer }, [pcm16.buffer]);
    }

    return true; // continuer
  }

  mixToMono_(a, b) {
    const len = Math.min(a.length, b.length);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = 0.5 * (a[i] + b[i]);
    return out;
  }

  totalLength_(arrs) {
    let n = 0;
    for (const a of arrs) n += a.length;
    return n;
  }

  drain_(arrs, target) {
    let offset = 0;
    while (arrs.length && offset < target.length) {
      const a = arrs[0];
      const need = target.length - offset;
      if (a.length <= need) {
        target.set(a, offset);
        offset += a.length;
        arrs.shift();
      } else {
        target.set(a.subarray(0, need), offset);
        arrs[0] = a.subarray(need);
        offset += need;
      }
    }
    // si on a légèrement plus que chunk, c'est OK : le reste reste en buffer
  }
}

registerProcessor('pcm16k-writer', PCM16kWriter);
