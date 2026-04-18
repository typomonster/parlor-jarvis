"use client";

import { useSyncExternalStore } from "react";

const WS_URL_KEY = "parlor.wsUrl";
const SYSTEM_PROMPT_KEY = "parlor.systemPrompt";
const VOICE_KEY = "parlor.voice";
const SPLIT_RATIO_KEY = "parlor.splitRatio";
const SPLIT_RATIO_DEFAULT = 0.4;
const SPLIT_RATIO_MIN = 0.2;
const SPLIT_RATIO_MAX = 0.75;

// Module-level store for user-configurable settings, same pattern as i18n.
let currentWsUrl: string | null = null;
const listeners = new Set<() => void>();

export function getDefaultWsUrl(): string {
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws`;
}

function readInitialWsUrl(): string {
  if (typeof window === "undefined") return "";
  const saved = window.localStorage.getItem(WS_URL_KEY);
  if (saved) return saved;
  return getDefaultWsUrl();
}

function getSnapshot(): string {
  if (currentWsUrl === null) currentWsUrl = readInitialWsUrl();
  return currentWsUrl;
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

function notify() {
  for (const cb of listeners) cb();
}

export function getWsUrl(): string {
  return getSnapshot();
}

export function setWsUrl(url: string) {
  currentWsUrl = url;
  if (typeof window !== "undefined") {
    window.localStorage.setItem(WS_URL_KEY, url);
  }
  notify();
}

export function resetWsUrl() {
  currentWsUrl = getDefaultWsUrl();
  if (typeof window !== "undefined") {
    window.localStorage.removeItem(WS_URL_KEY);
  }
  notify();
}

export function useWsUrl(): string {
  return useSyncExternalStore(subscribe, getSnapshot, () => "");
}

export function isWsUrlOverridden(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(WS_URL_KEY) !== null;
}

// ─────────────────────────────────────────────────────────────────────
// System prompt (shared chat column editor, persisted to localStorage)
// ─────────────────────────────────────────────────────────────────────

let currentSystemPrompt: string | null = null;
const promptListeners = new Set<() => void>();

function readInitialSystemPrompt(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(SYSTEM_PROMPT_KEY) ?? "";
}

function getSystemPromptSnapshot(): string {
  if (currentSystemPrompt === null) {
    currentSystemPrompt = readInitialSystemPrompt();
  }
  return currentSystemPrompt;
}

function subscribeSystemPrompt(cb: () => void) {
  promptListeners.add(cb);
  return () => {
    promptListeners.delete(cb);
  };
}

export function getSystemPrompt(): string {
  return getSystemPromptSnapshot();
}

export function setSystemPrompt(value: string) {
  currentSystemPrompt = value;
  if (typeof window !== "undefined") {
    if (value) {
      window.localStorage.setItem(SYSTEM_PROMPT_KEY, value);
    } else {
      window.localStorage.removeItem(SYSTEM_PROMPT_KEY);
    }
  }
  for (const cb of promptListeners) cb();
}

export function useSystemPrompt(): string {
  return useSyncExternalStore(
    subscribeSystemPrompt,
    getSystemPromptSnapshot,
    () => "",
  );
}

// ─────────────────────────────────────────────────────────────────────
// TTS voice preset — empty string means "backend default".
// ─────────────────────────────────────────────────────────────────────

let currentVoice: string | null = null;
const voiceListeners = new Set<() => void>();

function readInitialVoice(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(VOICE_KEY) ?? "";
}

function getVoiceSnapshot(): string {
  if (currentVoice === null) currentVoice = readInitialVoice();
  return currentVoice;
}

function subscribeVoice(cb: () => void) {
  voiceListeners.add(cb);
  return () => {
    voiceListeners.delete(cb);
  };
}

export function getVoice(): string {
  return getVoiceSnapshot();
}

export function setVoice(value: string) {
  currentVoice = value;
  if (typeof window !== "undefined") {
    if (value) window.localStorage.setItem(VOICE_KEY, value);
    else window.localStorage.removeItem(VOICE_KEY);
  }
  for (const cb of voiceListeners) cb();
}

export function useVoice(): string {
  return useSyncExternalStore(subscribeVoice, getVoiceSnapshot, () => "");
}

// Chat / sources column split ratio. SSR returns the default so the
// localStorage-backed client value doesn't trip a hydration mismatch.

let currentSplitRatio: number | null = null;
const splitListeners = new Set<() => void>();

function clampRatio(v: number): number {
  if (!Number.isFinite(v)) return SPLIT_RATIO_DEFAULT;
  return Math.min(SPLIT_RATIO_MAX, Math.max(SPLIT_RATIO_MIN, v));
}

function readInitialSplitRatio(): number {
  if (typeof window === "undefined") return SPLIT_RATIO_DEFAULT;
  const raw = window.localStorage.getItem(SPLIT_RATIO_KEY);
  if (raw === null) return SPLIT_RATIO_DEFAULT;
  return clampRatio(Number(raw));
}

function getSplitRatioSnapshot(): number {
  if (currentSplitRatio === null) currentSplitRatio = readInitialSplitRatio();
  return currentSplitRatio;
}

function subscribeSplitRatio(cb: () => void) {
  splitListeners.add(cb);
  return () => {
    splitListeners.delete(cb);
  };
}

export function setSplitRatio(v: number) {
  currentSplitRatio = clampRatio(v);
  if (typeof window !== "undefined") {
    window.localStorage.setItem(SPLIT_RATIO_KEY, String(currentSplitRatio));
  }
  for (const cb of splitListeners) cb();
}

export function useSplitRatio(): number {
  return useSyncExternalStore(
    subscribeSplitRatio,
    getSplitRatioSnapshot,
    () => SPLIT_RATIO_DEFAULT,
  );
}
