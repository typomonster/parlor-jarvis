"use client";

import { useCallback, useEffect, useRef } from "react";

type Props = {
  /** Current ratio for the LEFT column (0.15..0.85). */
  ratio: number;
  /** Called with the new left-column ratio while dragging. */
  onChange: (next: number) => void;
  /** Parent element whose width defines the track. */
  containerRef: React.RefObject<HTMLElement | null>;
  /** Optional ratio bounds. */
  min?: number;
  max?: number;
};

// Vertical draggable bar that lives between the chat column and the
// sources grid. Uses plain pointer events + requestAnimationFrame so we
// don't pull in a resize-panel library.
export function SplitDivider({
  ratio,
  onChange,
  containerRef,
  min = 0.2,
  max = 0.75,
}: Props) {
  const draggingRef = useRef(false);
  const rafRef = useRef<number | null>(null);

  const cleanupRef = useRef<(() => void) | null>(null);

  const onPointerMove = useCallback(
    (e: PointerEvent) => {
      if (!draggingRef.current) return;
      const container = containerRef.current;
      if (!container) return;
      if (rafRef.current !== null) return;
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null;
        const rect = container.getBoundingClientRect();
        const raw = (e.clientX - rect.left) / rect.width;
        const clamped = Math.min(max, Math.max(min, raw));
        onChange(clamped);
      });
    },
    [containerRef, max, min, onChange],
  );

  // Cleanup runs on unmount; while dragging, cleanupRef holds the
  // listener-teardown fn the pointerdown handler installed.
  useEffect(() => {
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      cleanupRef.current?.();
      cleanupRef.current = null;
    };
  }, []);

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    draggingRef.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    function endDrag() {
      draggingRef.current = false;
      document.body.style.removeProperty("cursor");
      document.body.style.removeProperty("user-select");
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", endDrag);
      window.removeEventListener("pointercancel", endDrag);
      cleanupRef.current = null;
    }

    cleanupRef.current = endDrag;
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", endDrag);
    window.addEventListener("pointercancel", endDrag);
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-valuenow={Math.round(ratio * 100)}
      aria-valuemin={Math.round(min * 100)}
      aria-valuemax={Math.round(max * 100)}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onKeyDown={(e) => {
        const step = 0.02;
        if (e.key === "ArrowLeft") onChange(Math.max(min, ratio - step));
        else if (e.key === "ArrowRight") onChange(Math.min(max, ratio + step));
      }}
      className="parlor-split-divider"
    />
  );
}
