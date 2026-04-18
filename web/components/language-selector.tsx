"use client";

import { Languages } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { LOCALES, LOCALE_LABELS, type Locale } from "@/lib/i18n/messages";
import { useI18n } from "@/lib/i18n/provider";

export function LanguageSelector() {
  const { locale, setLocale, t } = useI18n();

  return (
    <Select
      value={locale}
      onValueChange={(v) => {
        if (v) setLocale(v as Locale);
      }}
    >
      <SelectTrigger
        size="sm"
        aria-label={t("language.label")}
        className="min-w-[110px]"
      >
        <Languages className="size-3.5 text-muted-foreground" />
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="start">
        {LOCALES.map((l) => (
          <SelectItem key={l} value={l}>
            {LOCALE_LABELS[l]}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
