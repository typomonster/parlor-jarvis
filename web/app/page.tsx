"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  FileText,
  Film,
  Lock,
  Monitor,
  Video,
  X,
} from "lucide-react";

// lucide-react dropped its Github icon in recent versions; inline the logo
// so we don't pull in an extra dependency just for one glyph.
function GithubIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden="true"
      {...props}
    >
      <path d="M12 .5C5.73.5.5 5.73.5 12c0 5.08 3.29 9.39 7.86 10.92.58.11.79-.25.79-.56 0-.28-.01-1.01-.02-1.99-3.2.69-3.87-1.54-3.87-1.54-.52-1.33-1.28-1.68-1.28-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.1-.12-.29-.52-1.47.11-3.07 0 0 .97-.31 3.18 1.18a11.02 11.02 0 0 1 5.79 0c2.21-1.49 3.18-1.18 3.18-1.18.63 1.6.23 2.78.11 3.07.74.81 1.19 1.84 1.19 3.1 0 4.42-2.69 5.4-5.25 5.68.41.36.77 1.06.77 2.14 0 1.54-.01 2.79-.01 3.17 0 .31.21.67.8.56C20.71 21.39 24 17.08 24 12 24 5.73 18.27.5 12 .5Z" />
    </svg>
  );
}
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { FileUploadSkeleton } from "@/components/file-upload-skeleton";
import { SplitDivider } from "@/components/split-divider";
import { LanguageSelector } from "@/components/language-selector";
import { SettingsDialog } from "@/components/settings-dialog";
import { SystemPromptEditor } from "@/components/system-prompt-editor";
import { ThemeToggle } from "@/components/theme-toggle";
import { VoiceSelector } from "@/components/voice-selector";
import { useI18n } from "@/lib/i18n/provider";
import {
  getSystemPrompt,
  getVoice,
  getWsUrl,
  useSystemPrompt,
  useWsUrl,
} from "@/lib/settings";
import { cn } from "@/lib/utils";

type MachineState = "loading" | "listening" | "processing" | "speaking";
type ConnectionStatus = "connected" | "disconnected" | "processing";
type ImageSource = "camera" | "screen" | "pdf" | "video";

type Message =
  | {
      id: number;
      role: "user";
      pending: boolean;
      text: string;
      sources: ImageSource[];
    }
  | {
      id: number;
      role: "assistant";
      text: string;
      llmTime: number;
      ttsTime?: number;
    };

const STATE_COLORS: Record<MachineState, [string, string]> = {
  listening: ["#4ade80", "rgba(74,222,128,0.12)"],
  processing: ["#f59e0b", "rgba(245,158,11,0.12)"],
  speaking: ["#818cf8", "rgba(129,140,248,0.12)"],
  loading: ["#3a3d46", "rgba(58,61,70,0.12)"],
};

const STATE_LABEL_KEYS: Record<MachineState, string> = {
  loading: "state.loading",
  listening: "state.listening",
  processing: "state.thinking",
  speaking: "state.speaking",
};

const BAR_COUNT = 16;
const BAR_GAP = 2;
const BARGE_IN_GRACE_MS = 800;

// VAD + ONNX Runtime WASM assets are served from jsDelivr so we don't have to
// copy worklets/WASM into /public and keep them version-pinned manually.
const VAD_BASE_ASSET_PATH =
  "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.30/dist/";
const ONNX_WASM_BASE_PATH =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/";

function float32ToWavBase64(samples: Float32Array): string {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o: number, s: string) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  w(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true);
  v.setUint32(28, 32000, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

export default function Home() {
  const { t, locale } = useI18n();
  const activeWsUrl = useWsUrl();
  const activeSystemPrompt = useSystemPrompt();
  const [machineState, setMachineStateRender] =
    useState<MachineState>("loading");
  const [connection, setConnection] =
    useState<ConnectionStatus>("disconnected");
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [screenEnabled, setScreenEnabled] = useState(false);
  const [screenSending, setScreenSending] = useState(false);
  const [screenError, setScreenError] = useState<string | null>(null);
  const [pdfFileName, setPdfFileName] = useState<string | null>(null);
  const [pdfPage, setPdfPage] = useState(1);
  const [pdfPageCount, setPdfPageCount] = useState(0);
  const [pdfSending, setPdfSending] = useState(false);
  const [videoFileName, setVideoFileName] = useState<string | null>(null);
  const [videoSending, setVideoSending] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [backendUnreachable, setBackendUnreachable] = useState(false);
  const [splitRatio, setSplitRatio] = useState<number>(() => {
    if (typeof window === "undefined") return 0.4;
    const saved = Number(window.localStorage.getItem("parlor.splitRatio"));
    return Number.isFinite(saved) && saved >= 0.2 && saved <= 0.75
      ? saved
      : 0.4;
  });

  const cameraVideoRef = useRef<HTMLVideoElement | null>(null);
  const screenVideoRef = useRef<HTMLVideoElement | null>(null);
  const pdfCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoFileElRef = useRef<HTMLVideoElement | null>(null);
  const waveformCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);
  const transcriptRef = useRef<HTMLDivElement | null>(null);

  const machineStateRef = useRef<MachineState>("loading");
  const cameraEnabledRef = useRef<boolean>(true);
  const wsRef = useRef<WebSocket | null>(null);
  const wsShouldReconnectRef = useRef(true);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const screenEnabledRef = useRef(false);
  const screenSendingRef = useRef(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pdfDocRef = useRef<any>(null);
  const pdfLoadedRef = useRef(false);
  const pdfSendingRef = useRef(false);
  const videoLoadedRef = useRef(false);
  const videoSendingRef = useRef(false);
  const videoObjectUrlRef = useRef<string | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const micSourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamSourcesRef = useRef<AudioBufferSourceNode[]>([]);
  const streamNextTimeRef = useRef(0);
  const streamSampleRateRef = useRef(24000);
  const ignoreIncomingAudioRef = useRef(false);
  const speakingStartedAtRef = useRef(0);
  const waveformRAFRef = useRef<number | null>(null);
  const ambientPhaseRef = useRef(0);
  const msgIdRef = useRef(0);
  const localeRef = useRef<string>(locale);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const myvadRef = useRef<any>(null);
  const connectWsRef = useRef<() => void>(() => {});
  const wsHasConnectedRef = useRef(false);
  const wsFailedAttemptsRef = useRef(0);

  const nextMsgId = () => ++msgIdRef.current;

  const updateStateVars = useCallback((s: MachineState) => {
    const [glow, glowDim] = STATE_COLORS[s];
    document.documentElement.style.setProperty("--glow", glow);
    document.documentElement.style.setProperty("--glow-dim", glowDim);
  }, []);

  const setMachineState = useCallback(
    (s: MachineState) => {
      machineStateRef.current = s;
      setMachineStateRender(s);
      updateStateVars(s);

      if (myvadRef.current) {
        myvadRef.current.setOptions({
          positiveSpeechThreshold: s === "speaking" ? 0.92 : 0.5,
        });
      }

      const audioCtx = audioCtxRef.current;
      const analyser = analyserRef.current;
      const stream = cameraStreamRef.current;
      if (s === "listening" && stream && audioCtx && analyser) {
        if (!micSourceRef.current) {
          micSourceRef.current = audioCtx.createMediaStreamSource(stream);
        }
        try {
          micSourceRef.current.connect(analyser);
        } catch {}
      } else if (micSourceRef.current && s !== "listening" && analyser) {
        try {
          micSourceRef.current.disconnect(analyser);
        } catch {}
      }
    },
    [updateStateVars],
  );

  const ensureAudioCtx = useCallback(() => {
    if (!audioCtxRef.current) {
      const AC =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      audioCtxRef.current = new AC();
      analyserRef.current = audioCtxRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      analyserRef.current.smoothingTimeConstant = 0.75;
    }
  }, []);

  const stopPlayback = useCallback(() => {
    for (const src of streamSourcesRef.current) {
      try {
        src.stop();
      } catch {}
    }
    streamSourcesRef.current = [];
    streamNextTimeRef.current = 0;
  }, []);

  const startStreamPlayback = useCallback(() => {
    stopPlayback();
    ensureAudioCtx();
    const ctx = audioCtxRef.current!;
    if (ctx.state === "suspended") ctx.resume();
    streamNextTimeRef.current = ctx.currentTime + 0.05;
    speakingStartedAtRef.current = Date.now();
    setMachineState("speaking");
  }, [stopPlayback, ensureAudioCtx, setMachineState]);

  const queueAudioChunk = useCallback(
    (base64Pcm: string) => {
      ensureAudioCtx();
      const ctx = audioCtxRef.current!;
      const analyser = analyserRef.current!;

      const bin = atob(base64Pcm);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const int16 = new Int16Array(bytes.buffer);
      const float32 = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

      const audioBuffer = ctx.createBuffer(
        1,
        float32.length,
        streamSampleRateRef.current,
      );
      audioBuffer.getChannelData(0).set(float32);

      const source = ctx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(ctx.destination);
      source.connect(analyser);

      const startAt = Math.max(streamNextTimeRef.current, ctx.currentTime);
      source.start(startAt);
      streamNextTimeRef.current = startAt + audioBuffer.duration;

      streamSourcesRef.current.push(source);

      source.onended = () => {
        const idx = streamSourcesRef.current.indexOf(source);
        if (idx !== -1) streamSourcesRef.current.splice(idx, 1);
        if (
          streamSourcesRef.current.length === 0 &&
          machineStateRef.current === "speaking"
        ) {
          setMachineState("listening");
          setConnection("connected");
        }
      };
    },
    [ensureAudioCtx, setMachineState],
  );

  const captureCameraFrame = useCallback((): string | null => {
    const video = cameraVideoRef.current;
    if (!cameraEnabledRef.current || !video || !video.videoWidth) return null;
    const canvas = document.createElement("canvas");
    const scale = 320 / video.videoWidth;
    canvas.width = 320;
    canvas.height = video.videoHeight * scale;
    canvas
      .getContext("2d")!
      .drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
  }, []);

  // Screen captures tend to include text / UI, so we sample at a higher
  // resolution than the camera frame.
  const captureScreenFrame = useCallback((): string | null => {
    const video = screenVideoRef.current;
    if (
      !screenEnabledRef.current ||
      !screenSendingRef.current ||
      !video ||
      !video.videoWidth
    )
      return null;
    const canvas = document.createElement("canvas");
    const maxW = 768;
    const scale = Math.min(1, maxW / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas
      .getContext("2d")!
      .drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.75).split(",")[1];
  }, []);

  const disableScreenShare = useCallback(() => {
    screenStreamRef.current?.getTracks().forEach((t) => t.stop());
    screenStreamRef.current = null;
    if (screenVideoRef.current) screenVideoRef.current.srcObject = null;
    screenEnabledRef.current = false;
    screenSendingRef.current = false;
    setScreenEnabled(false);
    setScreenSending(false);
  }, []);

  const capturePdfFrame = useCallback((): string | null => {
    const source = pdfCanvasRef.current;
    if (
      !pdfLoadedRef.current ||
      !pdfSendingRef.current ||
      !source ||
      !source.width
    )
      return null;
    const maxW = 1024;
    const scale = Math.min(1, maxW / source.width);
    const out = document.createElement("canvas");
    out.width = Math.max(1, Math.round(source.width * scale));
    out.height = Math.max(1, Math.round(source.height * scale));
    out.getContext("2d")!.drawImage(source, 0, 0, out.width, out.height);
    return out.toDataURL("image/jpeg", 0.8).split(",")[1];
  }, []);

  const captureVideoFrame = useCallback((): string | null => {
    const video = videoFileElRef.current;
    if (
      !videoLoadedRef.current ||
      !videoSendingRef.current ||
      !video ||
      !video.videoWidth
    )
      return null;
    const canvas = document.createElement("canvas");
    const maxW = 640;
    const scale = Math.min(1, maxW / video.videoWidth);
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas
      .getContext("2d")!
      .drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.75).split(",")[1];
  }, []);

  const renderPdfPage = useCallback(async (pageNum: number) => {
    const doc = pdfDocRef.current;
    const canvas = pdfCanvasRef.current;
    if (!doc || !canvas) return;
    const page = await doc.getPage(pageNum);
    const viewport = page.getViewport({ scale: 1.5 });
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    await page.render({ canvasContext: ctx, canvas, viewport }).promise;
  }, []);

  const loadPdfFile = useCallback(
    async (file: File) => {
      const pdfjs = await import("pdfjs-dist");
      if (!pdfjs.GlobalWorkerOptions.workerSrc) {
        pdfjs.GlobalWorkerOptions.workerSrc = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;
      }
      const bytes = await file.arrayBuffer();
      const doc = await pdfjs.getDocument({ data: bytes }).promise;
      pdfDocRef.current = doc;
      pdfLoadedRef.current = true;
      pdfSendingRef.current = true;
      setPdfFileName(file.name);
      setPdfPageCount(doc.numPages);
      setPdfPage(1);
      setPdfSending(true);
      await renderPdfPage(1);
    },
    [renderPdfPage],
  );

  const removePdf = useCallback(() => {
    pdfDocRef.current?.destroy?.();
    pdfDocRef.current = null;
    pdfLoadedRef.current = false;
    pdfSendingRef.current = false;
    setPdfFileName(null);
    setPdfPageCount(0);
    setPdfPage(1);
    setPdfSending(false);
    const canvas = pdfCanvasRef.current;
    if (canvas) {
      canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
      canvas.width = 0;
      canvas.height = 0;
    }
  }, []);

  const changePdfPage = useCallback(
    (delta: number) => {
      setPdfPage((p) => {
        const next = Math.min(pdfPageCount, Math.max(1, p + delta));
        if (next !== p) void renderPdfPage(next);
        return next;
      });
    },
    [pdfPageCount, renderPdfPage],
  );

  const loadVideoFile = useCallback((file: File) => {
    if (videoObjectUrlRef.current) {
      URL.revokeObjectURL(videoObjectUrlRef.current);
    }
    const url = URL.createObjectURL(file);
    videoObjectUrlRef.current = url;
    if (videoFileElRef.current) {
      videoFileElRef.current.src = url;
    }
    videoLoadedRef.current = true;
    videoSendingRef.current = true;
    setVideoFileName(file.name);
    setVideoSending(true);
  }, []);

  const removeVideo = useCallback(() => {
    if (videoFileElRef.current) {
      try {
        videoFileElRef.current.pause();
      } catch {}
      videoFileElRef.current.removeAttribute("src");
      videoFileElRef.current.load();
    }
    if (videoObjectUrlRef.current) {
      URL.revokeObjectURL(videoObjectUrlRef.current);
      videoObjectUrlRef.current = null;
    }
    videoLoadedRef.current = false;
    videoSendingRef.current = false;
    setVideoFileName(null);
    setVideoSending(false);
  }, []);

  const enableScreenShare = useCallback(async (): Promise<boolean> => {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      });
      screenStreamRef.current = stream;
      if (screenVideoRef.current) screenVideoRef.current.srcObject = stream;
      screenEnabledRef.current = true;
      screenSendingRef.current = true;
      setScreenEnabled(true);
      setScreenSending(true);
      setScreenError(null);
      // If the user stops sharing via the browser UI, clean up our state.
      const track = stream.getVideoTracks()[0];
      if (track) {
        track.addEventListener("ended", () => disableScreenShare());
      }
      return true;
    } catch (e) {
      console.warn("Screen share failed:", (e as Error).message);
      return false;
    }
  }, [disableScreenShare]);

  const startWaveformLoop = useCallback(() => {
    function tick() {
      const canvas = waveformCanvasRef.current;
      if (!canvas) return;
      const ctx2d = canvas.getContext("2d");
      if (!ctx2d) return;
      const w = canvas.getBoundingClientRect().width;
      const h = canvas.getBoundingClientRect().height;
      ctx2d.clearRect(0, 0, w, h);

      const barWidth = (w - (BAR_COUNT - 1) * BAR_GAP) / BAR_COUNT;
      const [glow] = STATE_COLORS[machineStateRef.current];
      ctx2d.fillStyle = glow;

      let dataArray: Uint8Array<ArrayBuffer> | null = null;
      if (analyserRef.current) {
        dataArray = new Uint8Array(
          new ArrayBuffer(analyserRef.current.frequencyBinCount),
        );
        analyserRef.current.getByteFrequencyData(dataArray);
      }

      for (let i = 0; i < BAR_COUNT; i++) {
        let amplitude = 0;
        if (dataArray) {
          const binIndex = Math.floor((i / BAR_COUNT) * dataArray.length * 0.6);
          amplitude = dataArray[binIndex] / 255;
        }
        if (!dataArray || amplitude < 0.02) {
          ambientPhaseRef.current += 0.0001;
          const drift =
            Math.sin(ambientPhaseRef.current * 3 + i * 0.4) * 0.5 + 0.5;
          amplitude = 0.03 + drift * 0.04;
        }

        const barH = Math.max(2, amplitude * (h - 4));
        const x = i * (barWidth + BAR_GAP);
        const y = (h - barH) / 2;

        ctx2d.globalAlpha = 0.3 + amplitude * 0.7;
        ctx2d.beginPath();
        const r = Math.min(barWidth / 2, barH / 2, 3);
        ctx2d.roundRect(x, y, barWidth, barH, r);
        ctx2d.fill();
      }
      ctx2d.globalAlpha = 1;
      waveformRAFRef.current = requestAnimationFrame(tick);
    }
    tick();
  }, []);

  const initWaveformCanvas = useCallback(() => {
    const canvas = waveformCanvasRef.current;
    if (!canvas) return;
    const ctx2d = canvas.getContext("2d");
    if (!ctx2d) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx2d.scale(dpr, dpr);
  }, []);

  const handleSpeechStart = useCallback(() => {
    if (machineStateRef.current === "speaking") {
      if (Date.now() - speakingStartedAtRef.current < BARGE_IN_GRACE_MS) {
        console.log("Barge-in suppressed (echo grace period)");
        return;
      }
      stopPlayback();
      ignoreIncomingAudioRef.current = true;
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "interrupt" }));
      }
      setMachineState("listening");
      console.log("Barge-in: interrupted playback");
    }
  }, [stopPlayback, setMachineState]);

  const handleSpeechEnd = useCallback(
    (audio: Float32Array) => {
      if (machineStateRef.current !== "listening") return;
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      const wavBase64 = float32ToWavBase64(audio);
      const imageBase64 = captureCameraFrame();
      const screenBase64 = captureScreenFrame();
      const pdfBase64 = capturePdfFrame();
      const videoBase64 = captureVideoFrame();

      setMachineState("processing");
      setConnection("processing");
      const usedSources: ImageSource[] = [];
      if (imageBase64) usedSources.push("camera");
      if (screenBase64) usedSources.push("screen");
      if (pdfBase64) usedSources.push("pdf");
      if (videoBase64) usedSources.push("video");

      setMessages((prev) => [
        ...prev,
        {
          id: nextMsgId(),
          role: "user",
          pending: true,
          text: "",
          sources: usedSources,
        },
      ]);

      // Batch every active source into a single list — the backend walks
      // it in order, appending each blob as an image content part.
      const images: { source: ImageSource; blob: string }[] = [];
      if (imageBase64) images.push({ source: "camera", blob: imageBase64 });
      if (screenBase64) images.push({ source: "screen", blob: screenBase64 });
      if (pdfBase64) images.push({ source: "pdf", blob: pdfBase64 });
      if (videoBase64) images.push({ source: "video", blob: videoBase64 });

      ws.send(
        JSON.stringify({
          audio: wavBase64,
          images,
          lang: localeRef.current,
          voice: getVoice(),
          system_prompt: getSystemPrompt(),
        }),
      );
    },
    [
      captureCameraFrame,
      captureScreenFrame,
      capturePdfFrame,
      captureVideoFrame,
      setMachineState,
    ],
  );

  const connectWs = useCallback(() => {
    const ws = new WebSocket(getWsUrl());
    wsRef.current = ws;

    ws.onopen = () => {
      wsHasConnectedRef.current = true;
      wsFailedAttemptsRef.current = 0;
      setBackendUnreachable(false);
      setConnection("connected");
      if (machineStateRef.current !== "loading") setMachineState("listening");
    };

    ws.onerror = () => {
      // Browsers don't expose details; rely on onclose for the actual message.
      wsFailedAttemptsRef.current += 1;
    };

    ws.onclose = (ev) => {
      setConnection("disconnected");
      const abnormal = ev.code !== 1000 && ev.code !== 1001;
      if (!wsHasConnectedRef.current && abnormal) {
        setBackendUnreachable(true);
      }
      if (wsShouldReconnectRef.current) {
        setTimeout(() => connectWsRef.current(), 2000);
      }
    };

    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data);
      if (msg.type === "text") {
        setMessages((prev) => {
          const next = [...prev];
          if (msg.transcription) {
            for (let i = next.length - 1; i >= 0; i--) {
              const m = next[i];
              if (m.role === "user" && m.pending) {
                next[i] = {
                  ...m,
                  pending: false,
                  text: msg.transcription as string,
                };
                break;
              }
            }
          }
          next.push({
            id: nextMsgId(),
            role: "assistant",
            text: msg.text as string,
            llmTime: msg.llm_time as number,
          });
          return next;
        });
      } else if (msg.type === "audio_start") {
        if (ignoreIncomingAudioRef.current) return;
        streamSampleRateRef.current = msg.sample_rate || 24000;
        startStreamPlayback();
      } else if (msg.type === "audio_chunk") {
        if (ignoreIncomingAudioRef.current) return;
        queueAudioChunk(msg.audio);
      } else if (msg.type === "audio_end") {
        if (ignoreIncomingAudioRef.current) {
          ignoreIncomingAudioRef.current = false;
          stopPlayback();
          setMachineState("listening");
          return;
        }
        setMessages((prev) => {
          const next = [...prev];
          for (let i = next.length - 1; i >= 0; i--) {
            const m = next[i];
            if (m.role === "assistant") {
              next[i] = { ...m, ttsTime: msg.tts_time as number };
              break;
            }
          }
          return next;
        });
      }
    };
  }, [queueAudioChunk, setMachineState, startStreamPlayback, stopPlayback]);

  useEffect(() => {
    connectWsRef.current = connectWs;
  }, [connectWs]);

  useEffect(() => {
    window.localStorage.setItem("parlor.splitRatio", String(splitRatio));
  }, [splitRatio]);

  // Keep latest locale reachable from memoized handlers so server-side
  // Supertonic speaks in whatever language the UI is set to, even after
  // VAD's onSpeechEnd closure was created.
  useEffect(() => {
    localeRef.current = locale;
  }, [locale]);

  // Reconnect when the user changes the WebSocket URL in Settings.
  // Skip the first render — the mount effect handles the initial connect.
  const prevWsUrlRef = useRef<string | null>(null);
  useEffect(() => {
    if (prevWsUrlRef.current === null) {
      prevWsUrlRef.current = activeWsUrl;
      return;
    }
    if (prevWsUrlRef.current === activeWsUrl) return;
    prevWsUrlRef.current = activeWsUrl;

    // Reset error state so the banner doesn't linger from the old URL,
    // then close the current socket. onclose will schedule a reconnect
    // against the new URL via connectWsRef.current().
    wsHasConnectedRef.current = false;
    setBackendUnreachable(false);
    try {
      wsRef.current?.close();
    } catch {}
  }, [activeWsUrl]);

  // litert-lm only supports one conversation per WS connection, so when
  // the user edits the system prompt we close the socket — the existing
  // auto-reconnect loop reopens it and the first message on the new
  // connection carries the new prompt into conversation creation.
  const prevSystemPromptRef = useRef<string | null>(null);
  useEffect(() => {
    if (prevSystemPromptRef.current === null) {
      prevSystemPromptRef.current = activeSystemPrompt;
      return;
    }
    if (prevSystemPromptRef.current === activeSystemPrompt) return;
    prevSystemPromptRef.current = activeSystemPrompt;
    try {
      wsRef.current?.close();
    } catch {}
  }, [activeSystemPrompt]);

  const startCamera = useCallback(async (): Promise<void> => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      cameraStreamRef.current = stream;
      if (cameraVideoRef.current) cameraVideoRef.current.srcObject = stream;
      return;
    } catch (e) {
      console.warn("Video+audio failed:", (e as Error).message);
    }

    const results = await Promise.allSettled([
      navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      }),
      navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      }),
    ]);
    const combined = new MediaStream();
    for (const r of results) {
      if (r.status === "fulfilled") {
        r.value.getTracks().forEach((t) => combined.addTrack(t));
      }
    }
    cameraStreamRef.current = combined;
    if (combined.getVideoTracks().length && cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = combined;
    }
    if (!combined.getAudioTracks().length) {
      setCameraEnabled(false);
      cameraEnabledRef.current = false;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;

    initWaveformCanvas();
    const onResize = () => initWaveformCanvas();
    window.addEventListener("resize", onResize);

    updateStateVars("loading");

    (async () => {
      await startCamera();
      if (cancelled) return;

      connectWs();

      const { MicVAD } = await import("@ricky0123/vad-web");
      if (cancelled) return;

      myvadRef.current = await MicVAD.new({
        getStream: async () =>
          new MediaStream(cameraStreamRef.current!.getAudioTracks()),
        positiveSpeechThreshold: 0.5,
        negativeSpeechThreshold: 0.25,
        redemptionMs: 600,
        minSpeechMs: 300,
        preSpeechPadMs: 300,
        onSpeechStart: handleSpeechStart,
        onSpeechEnd: handleSpeechEnd,
        onVADMisfire: () => console.log("VAD misfire (too short)"),
        onnxWASMBasePath: ONNX_WASM_BASE_PATH,
        baseAssetPath: VAD_BASE_ASSET_PATH,
      });
      if (cancelled) return;

      myvadRef.current.start();

      const initAudio = () => {
        ensureAudioCtx();
        const ctx = audioCtxRef.current;
        if (ctx?.state === "suspended") ctx.resume();
        document.removeEventListener("click", initAudio);
        document.removeEventListener("keydown", initAudio);
      };
      document.addEventListener("click", initAudio);
      document.addEventListener("keydown", initAudio);
      ensureAudioCtx();

      setMachineState("listening");
      startWaveformLoop();
      console.log("VAD initialized and listening");
    })().catch((e) => console.error("Init error:", e));

    return () => {
      cancelled = true;
      wsShouldReconnectRef.current = false;
      window.removeEventListener("resize", onResize);
      if (waveformRAFRef.current) cancelAnimationFrame(waveformRAFRef.current);
      try {
        myvadRef.current?.pause?.();
      } catch {}
      try {
        wsRef.current?.close();
      } catch {}
      stopPlayback();
      cameraStreamRef.current?.getTracks().forEach((t) => t.stop());
      screenStreamRef.current?.getTracks().forEach((t) => t.stop());
      pdfDocRef.current?.destroy?.();
      if (videoObjectUrlRef.current) {
        URL.revokeObjectURL(videoObjectUrlRef.current);
        videoObjectUrlRef.current = null;
      }
      try {
        audioCtxRef.current?.close();
      } catch {}
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const viewport = transcriptRef.current?.querySelector<HTMLElement>(
      '[data-slot="scroll-area-viewport"]',
    );
    if (viewport) viewport.scrollTop = viewport.scrollHeight;
  }, [messages]);

  const onCameraToggle = () => {
    const next = !cameraEnabledRef.current;
    cameraEnabledRef.current = next;
    setCameraEnabled(next);
    if (cameraVideoRef.current) cameraVideoRef.current.style.opacity = next ? "1" : "0.3";
  };

  // Start / stop the screen share entirely (used by skeleton button and
  // the bottom-right X chip when a share is active).
  const onScreenToggleShare = async () => {
    if (screenEnabledRef.current) {
      disableScreenShare();
      return;
    }
    const ok = await enableScreenShare();
    if (!ok) setScreenError(t("screen.permissionDenied"));
    else setScreenError(null);
  };

  // Toggle whether an active source actually sends frames to the model —
  // symmetric with the Camera On/Off chip.
  const onScreenSendToggle = () => {
    const next = !screenSendingRef.current;
    screenSendingRef.current = next;
    setScreenSending(next);
  };

  const onPdfSendToggle = () => {
    const next = !pdfSendingRef.current;
    pdfSendingRef.current = next;
    setPdfSending(next);
  };

  const onVideoSendToggle = () => {
    const next = !videoSendingRef.current;
    videoSendingRef.current = next;
    setVideoSending(next);
  };

  const connectionLabel =
    connection === "connected"
      ? t("status.connected")
      : connection === "processing"
        ? t("status.processing")
        : t("status.disconnected");

  const tileStateClass = (active: boolean) =>
    `viewport-wrap ${active ? machineState : "loading"}`;

  // Shared pill styling for every source tile's On/Off toggle. The "on"
  // state is a solid emerald chip so it reads clearly over live media.
  const sourceChipClass = (on: boolean) =>
    cn(
      "rounded-full px-3 text-[10px] font-semibold uppercase tracking-wider transition-colors",
      on
        ? "bg-emerald-500 text-white hover:bg-emerald-500/90 focus-visible:ring-emerald-500/50"
        : "bg-neutral-900/70 text-white/80 hover:bg-neutral-900/80 backdrop-blur-sm",
    );

  return (
    <div className="parlor-body">
      <nav className="parlor-navbar">
        <div className="parlor-navbar-left">
          <div className="parlor-logo">
            <div className="parlor-logo-mark" />
            <h1 className="text-[20px] font-semibold leading-none text-foreground">
              Parlor
            </h1>
            <Badge
              variant="outline"
              className="h-auto border-white/10 bg-transparent px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-[0.12em] text-foreground/50 dark:border-white/10"
            >
              {t("multilingual")}
            </Badge>
          </div>
        </div>

        <div className="parlor-navbar-center">
          <span className="parlor-model-label">Gemma 4 E2B</span>
          <LanguageSelector />
          <VoiceSelector />
        </div>

        <div className="parlor-navbar-right">
          <Badge
            variant="secondary"
            className={cn(
              "h-auto gap-2 rounded-full border-transparent px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.04em] transition-colors",
              connection === "connected" &&
                "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400",
              connection === "disconnected" &&
                "bg-red-500/10 text-red-600 dark:text-red-400",
              connection === "processing" &&
                "bg-amber-500/10 text-amber-600 dark:text-amber-400",
            )}
          >
            <span
              className={cn(
                "size-1.5 rounded-full transition-colors",
                connection === "connected" && "bg-emerald-500",
                connection === "disconnected" && "bg-red-500",
                connection === "processing" && "bg-amber-500",
              )}
            />
            {connectionLabel}
          </Badge>
          <ThemeToggle />
          <SettingsDialog />
        </div>
      </nav>

      {backendUnreachable ? (
        <div className="parlor-alert-wrap">
          <Alert variant="destructive">
            <AlertTitle>{t("backendError.title")}</AlertTitle>
            <AlertDescription>
              {t("backendError.body", { url: activeWsUrl })}
            </AlertDescription>
          </Alert>
        </div>
      ) : null}

      <main
        ref={mainRef}
        className="parlor-main"
        style={{
          gridTemplateColumns: `minmax(0, ${splitRatio}fr) auto minmax(0, ${
            1 - splitRatio
          }fr)`,
        }}
      >
        {/* Left column — AI conversation */}
        <section className="parlor-chat-column">
          <SystemPromptEditor />
          <div ref={transcriptRef} className="transcript-wrap">
            <ScrollArea className="transcript-scroll-area">
              <div className="transcript-content">
                {messages.map((m) =>
                  m.role === "user" ? (
                    <div key={m.id} className="msg user">
                      {m.pending ? (
                        <span className="loading-dots">
                          <span />
                          <span />
                          <span />
                        </span>
                      ) : (
                        m.text
                      )}
                      {m.sources.length > 0 ? (
                        <div className="meta">
                          {m.sources
                            .map((s) => t(`with.${s}`))
                            .join(" · ")}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div key={m.id} className="msg assistant">
                      {m.text}
                      <div className="meta">
                        {`LLM ${m.llmTime}s`}
                        {m.ttsTime !== undefined
                          ? ` · TTS ${m.ttsTime}s`
                          : null}
                      </div>
                    </div>
                  ),
                )}
              </div>
            </ScrollArea>
          </div>

          <div className="parlor-chat-footer">
            <div className="state-indicator">
              <div className={`dot ${machineState}`} />
              <span>{t(STATE_LABEL_KEYS[machineState])}</span>
              <canvas ref={waveformCanvasRef} className="state-wave" />
            </div>
            <div className="parlor-chat-footer-right">
              <Tooltip>
                <TooltipTrigger
                  render={
                    <Badge
                      variant="outline"
                      className="h-auto cursor-help gap-1.5 border-border/50 bg-transparent px-2 py-1 text-[10px] font-medium uppercase tracking-wider text-foreground/50"
                    >
                      <Lock className="size-2.5" strokeWidth={1.5} />
                      {t("onDevice")}
                    </Badge>
                  }
                />
                <TooltipContent className="max-w-[240px] text-xs leading-snug">
                  {t("onDevice.tooltip")}
                </TooltipContent>
              </Tooltip>
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="GitHub"
                className="text-foreground/50 hover:text-foreground"
                nativeButton={false}
                render={
                  <a
                    href="https://github.com/typomonster/parlor-multilingual"
                    target="_blank"
                    rel="noreferrer noopener"
                  />
                }
              >
                <GithubIcon />
              </Button>
            </div>
          </div>
        </section>

        <SplitDivider
          ratio={splitRatio}
          onChange={setSplitRatio}
          containerRef={mainRef}
        />

        {/* Right column — 2×2 sources grid */}
        <section className="parlor-sources-grid">
          {/* Camera tile (top-left): live from the start */}
          <div className={tileStateClass(cameraEnabled)}>
            <div className="tile-label">
              <Video className="size-3" />
              {t("tab.camera")}
            </div>
            <video
              ref={cameraVideoRef}
              autoPlay
              muted
              playsInline
              className="mirror"
            />
            <div className="tile-actions">
              <Button
                variant="ghost"
                size="xs"
                onClick={onCameraToggle}
                className={sourceChipClass(cameraEnabled)}
              >
                {cameraEnabled ? t("camera.on") : t("camera.off")}
              </Button>
            </div>
          </div>

          {/* Screen tile (top-right) */}
          <div className={tileStateClass(screenEnabled)}>
            <div className="tile-label">
              <Monitor className="size-3" />
              {t("tab.screen")}
            </div>
            <video
              ref={screenVideoRef}
              autoPlay
              muted
              playsInline
              className="screen-video"
              data-hidden={!screenEnabled}
            />
            {screenEnabled ? (
              <>
                <div className="tile-actions">
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={onScreenSendToggle}
                    className={sourceChipClass(screenSending)}
                  >
                    {screenSending ? t("screen.on") : t("screen.off")}
                  </Button>
                </div>
                <div className="pdf-pager">
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label={t("file.remove")}
                    onClick={onScreenToggleShare}
                  >
                    <X />
                  </Button>
                </div>
              </>
            ) : (
              <div className="viewport-upload">
                <p className="viewport-upload-title">{t("tab.screen")}</p>
                <Button variant="outline" size="sm" onClick={onScreenToggleShare}>
                  <Monitor />
                  {t("upload.screen")}
                </Button>
                {screenError ? (
                  <p className="text-destructive text-[11px]">{screenError}</p>
                ) : null}
              </div>
            )}
          </div>

          {/* PDF tile (bottom-left) */}
          <div className={tileStateClass(!!pdfFileName)}>
            <div className="tile-label">
              <FileText className="size-3" />
              {t("tab.pdf")}
            </div>
            <canvas
              ref={pdfCanvasRef}
              className="pdf-canvas"
              data-hidden={!pdfFileName}
            />
            {pdfFileName ? (
              <>
                <div className="tile-actions">
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={onPdfSendToggle}
                    className={sourceChipClass(pdfSending)}
                  >
                    {pdfSending ? t("pdf.on") : t("pdf.off")}
                  </Button>
                </div>
                {pdfPageCount > 0 ? (
                  <div className="pdf-pager">
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label={t("pdf.prev")}
                      disabled={pdfPage <= 1}
                      onClick={() => changePdfPage(-1)}
                    >
                      <ChevronLeft />
                    </Button>
                    <span>
                      {t("pdf.page", {
                        current: String(pdfPage),
                        total: String(pdfPageCount),
                      })}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label={t("pdf.next")}
                      disabled={pdfPage >= pdfPageCount}
                      onClick={() => changePdfPage(1)}
                    >
                      <ChevronRight />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      aria-label={t("file.remove")}
                      onClick={removePdf}
                      className="ml-1"
                    >
                      <X />
                    </Button>
                  </div>
                ) : null}
              </>
            ) : (
              <FileUploadSkeleton
                title={t("tab.pdf")}
                accept="application/pdf"
                buttonLabel={t("upload.pdf")}
                onFile={(f) => void loadPdfFile(f)}
              />
            )}
          </div>

          {/* Video tile (bottom-right) */}
          <div className={tileStateClass(!!videoFileName)}>
            <div className="tile-label">
              <Film className="size-3" />
              {t("tab.video")}
            </div>
            <video
              ref={videoFileElRef}
              controls
              playsInline
              className="screen-video uploaded-video"
              data-hidden={!videoFileName}
            />
            {videoFileName ? (
              <>
                <div className="tile-actions">
                  <Button
                    variant="ghost"
                    size="xs"
                    onClick={onVideoSendToggle}
                    className={sourceChipClass(videoSending)}
                  >
                    {videoSending ? t("video.on") : t("video.off")}
                  </Button>
                </div>
                <div className="pdf-pager">
                  <span className="video-file-name">{videoFileName}</span>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label={t("file.remove")}
                    onClick={removeVideo}
                    className="ml-1"
                  >
                    <X />
                  </Button>
                </div>
              </>
            ) : (
              <FileUploadSkeleton
                title={t("tab.video")}
                accept="video/*"
                buttonLabel={t("upload.video")}
                onFile={loadVideoFile}
              />
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
