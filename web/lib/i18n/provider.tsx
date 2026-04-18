"use client";

import { useCallback, useSyncExternalStore } from "react";
import {
  DEFAULT_LOCALE,
  isLocale,
  type Locale,
  MESSAGES,
} from "./messages";

const STORAGE_KEY = "parlor.locale";

// Module-level store. Keeps locale coherent across components without
// passing a Context down the tree.
let currentLocale: Locale | null = null;
const listeners = new Set<() => void>();

function readInitialLocale(): Locale {
  if (typeof window === "undefined") return DEFAULT_LOCALE;
  const saved = window.localStorage.getItem(STORAGE_KEY);
  if (saved && isLocale(saved)) return saved;
  const browser = navigator.language.slice(0, 2);
  return isLocale(browser) ? browser : DEFAULT_LOCALE;
}

function getSnapshot(): Locale {
  if (currentLocale === null) currentLocale = readInitialLocale();
  return currentLocale;
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => {
    listeners.delete(cb);
  };
}

export function setLocale(l: Locale) {
  currentLocale = l;
  if (typeof window !== "undefined") {
    window.localStorage.setItem(STORAGE_KEY, l);
    document.documentElement.lang = l;
  }
  for (const cb of listeners) cb();
}

export function I18nProvider({ children }: { children: React.ReactNode }) {
  // No Context needed — the module-level store handles subscriptions.
  return <>{children}</>;
}

export function useI18n() {
  const locale = useSyncExternalStore(
    subscribe,
    getSnapshot,
    () => DEFAULT_LOCALE,
  );

  const t = useCallback(
    (key: string, vars?: Record<string, string>) => {
      const dict = MESSAGES[locale] ?? MESSAGES[DEFAULT_LOCALE];
      let out = dict[key] ?? MESSAGES[DEFAULT_LOCALE][key] ?? key;
      if (vars) {
        for (const [k, v] of Object.entries(vars)) {
          out = out.replaceAll(`{${k}}`, v);
        }
      }
      return out;
    },
    [locale],
  );

  return { locale, setLocale, t };
}
