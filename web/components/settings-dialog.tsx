"use client";

import { useState } from "react";
import { Settings, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useI18n } from "@/lib/i18n/provider";
import {
  getDefaultWsUrl,
  resetWsUrl,
  setWsUrl,
  useWsUrl,
} from "@/lib/settings";

function isValidWsUrl(url: string): boolean {
  return /^wss?:\/\/.+/.test(url.trim());
}

export function SettingsDialog() {
  const { t } = useI18n();
  const activeUrl = useWsUrl();
  const [open, setOpen] = useState(false);
  const [draft, setDraft] = useState(activeUrl);
  const [error, setError] = useState<string | null>(null);

  // Reset the draft whenever the dialog opens, so cancel discards edits.
  const handleOpenChange = (next: boolean) => {
    if (next) {
      setDraft(activeUrl);
      setError(null);
    }
    setOpen(next);
  };

  const defaultUrl = getDefaultWsUrl();

  const onSave = () => {
    const next = draft.trim();
    if (!isValidWsUrl(next)) {
      setError(t("settings.invalid"));
      return;
    }
    setWsUrl(next);
    setOpen(false);
  };

  const onReset = () => {
    resetWsUrl();
    setDraft(getDefaultWsUrl());
    setError(null);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger
        render={
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label={t("settings.open")}
            className="text-foreground/60 hover:text-foreground"
          >
            <Settings />
          </Button>
        }
      />
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("settings.title")}</DialogTitle>
          <DialogDescription>{t("settings.description")}</DialogDescription>
        </DialogHeader>

        <div className="grid gap-2">
          <Label htmlFor="ws-url">{t("settings.wsUrl.label")}</Label>
          <Input
            id="ws-url"
            value={draft}
            placeholder={defaultUrl}
            onChange={(e) => {
              setDraft(e.target.value);
              if (error) setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                onSave();
              }
            }}
            autoComplete="off"
            spellCheck={false}
            aria-invalid={error !== null}
          />
          <p className="text-xs text-muted-foreground">
            {t("settings.wsUrl.help")}
          </p>
          <p className="text-xs text-muted-foreground/70">
            {t("settings.wsUrl.default", { url: defaultUrl })}
          </p>
          {error ? (
            <p className="text-xs text-destructive">{error}</p>
          ) : null}
        </div>

        <DialogFooter className="sm:justify-between">
          <Button variant="ghost" size="sm" onClick={onReset} type="button">
            <RotateCcw />
            {t("settings.reset")}
          </Button>
          <div className="flex gap-2">
            <DialogClose
              render={
                <Button variant="outline" size="sm" type="button">
                  {t("settings.cancel")}
                </Button>
              }
            />
            <Button size="sm" onClick={onSave} type="button">
              {t("settings.save")}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
