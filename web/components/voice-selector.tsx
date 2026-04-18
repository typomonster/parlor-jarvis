"use client";

import { Mic2 } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useI18n } from "@/lib/i18n/provider";
import { setVoice, useVoice } from "@/lib/settings";

// Supertonic-2 ships with M1–M5 and F1–F5. Kokoro uses different IDs
// ("af_heart" etc.) — picking a value the active backend doesn't
// recognise falls back to its own default voice on the server side.
const MALE_VOICES = ["M1", "M2", "M3", "M4", "M5"];
const FEMALE_VOICES = ["F1", "F2", "F3", "F4", "F5"];

export function VoiceSelector() {
  const { t } = useI18n();
  const voice = useVoice();

  return (
    <Select
      value={voice || "__default__"}
      onValueChange={(v) => {
        if (v === "__default__") setVoice("");
        else if (v) setVoice(v);
      }}
    >
      <SelectTrigger
        size="sm"
        aria-label={t("voice.label")}
        className="min-w-[96px]"
      >
        <Mic2 className="size-3.5 text-muted-foreground" />
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="start">
        <SelectItem value="__default__">{t("voice.default")}</SelectItem>
        <SelectSeparator />
        <SelectGroup>
          <SelectLabel>{t("voice.male")}</SelectLabel>
          {MALE_VOICES.map((v) => (
            <SelectItem key={v} value={v}>
              {v}
            </SelectItem>
          ))}
        </SelectGroup>
        <SelectSeparator />
        <SelectGroup>
          <SelectLabel>{t("voice.female")}</SelectLabel>
          {FEMALE_VOICES.map((v) => (
            <SelectItem key={v} value={v}>
              {v}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
