"use client";

import { useSyncExternalStore } from "react";

const WS_URL_KEY = "parlor.wsUrl";

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
