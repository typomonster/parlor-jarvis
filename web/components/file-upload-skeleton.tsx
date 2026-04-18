"use client";

import { useRef } from "react";
import { Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/lib/i18n/provider";

type Props = {
  /** Uppercase label rendered at the top of the skeleton tile. */
  title: string;
  /** `accept` attribute forwarded to the hidden file input. */
  accept: string;
  /** Primary CTA label (e.g. "Upload PDF"). */
  buttonLabel: string;
  /** Called with the selected file once the user picks one. */
  onFile: (file: File) => void;
};

// Shared empty-state for tiles backed by a user-uploaded file (PDF / video).
// Renders the hatched background, a title, a file-picker button, and the
// localised "Click to choose a file" hint.
export function FileUploadSkeleton({
  title,
  accept,
  buttonLabel,
  onFile,
}: Props) {
  const { t } = useI18n();
  const inputRef = useRef<HTMLInputElement | null>(null);

  return (
    <div className="viewport-upload">
      <p className="viewport-upload-title">{title}</p>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
      <Button
        variant="outline"
        size="sm"
        onClick={() => inputRef.current?.click()}
      >
        <Upload />
        {buttonLabel}
      </Button>
      <p>{t("upload.hint")}</p>
    </div>
  );
}
