"use client";

import { useSyncExternalStore } from "react";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n/provider";

// Returns false during SSR / before hydration, true on the client.
// Avoids the "icon flips on mount" flash and hydration mismatch.
function useIsClient() {
  return useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );
}

export function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const { t } = useI18n();
  const isClient = useIsClient();
  const isDark = isClient && resolvedTheme === "dark";

  return (
    <Button
      variant="ghost"
      size="icon-sm"
      aria-label={t("theme.toggle")}
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="text-foreground/60 hover:text-foreground"
    >
      {isDark ? <Moon /> : <Sun />}
    </Button>
  );
}
