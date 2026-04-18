"use client";

import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useI18n } from "@/lib/i18n/provider";
import { setSystemPrompt, useSystemPrompt } from "@/lib/settings";

// Lives at the top of the chat column. Empty value → server uses the
// default SYSTEM_PROMPT. Anything else is hot-swapped into the
// conversation by the backend on the next speech-end message (which
// resets conversation history — by design).
export function SystemPromptEditor() {
  const { t } = useI18n();
  const value = useSystemPrompt();

  return (
    <div className="parlor-system-prompt">
      <div className="parlor-system-prompt-header">
        <Label
          htmlFor="parlor-system-prompt"
          className="text-[10px] font-medium uppercase tracking-wider text-foreground/50"
        >
          {t("systemPrompt.label")}
        </Label>
        {value ? (
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => setSystemPrompt("")}
            className="h-auto gap-1 px-1.5 py-0 text-[10px] text-foreground/50 hover:text-foreground"
          >
            <RotateCcw className="size-3" />
            {t("systemPrompt.reset")}
          </Button>
        ) : null}
      </div>
      <Textarea
        id="parlor-system-prompt"
        value={value}
        onChange={(e) => setSystemPrompt(e.target.value)}
        placeholder={t("systemPrompt.placeholder")}
        rows={2}
        className="min-h-[48px] resize-y text-xs leading-snug"
      />
    </div>
  );
}
