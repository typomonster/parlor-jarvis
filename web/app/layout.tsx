import type { Metadata } from "next";
import { Instrument_Sans, Noto_Sans_KR, Syne } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import { I18nProvider } from "@/lib/i18n/provider";
import "./globals.css";

const instrumentSans = Instrument_Sans({
  variable: "--font-latin",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

// Noto Sans KR covers Korean glyphs; browsers will fall through to it for
// characters that Instrument Sans can't render. Also reasonable coverage for
// other CJK punctuation / ambiguous glyphs.
const notoSansKR = Noto_Sans_KR({
  variable: "--font-cjk",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const syne = Syne({
  variable: "--font-display-latin",
  subsets: ["latin"],
  weight: ["700", "800"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Parlor",
  description: "On-device, real-time multimodal AI",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${instrumentSans.variable} ${notoSansKR.variable} ${syne.variable}`}
      suppressHydrationWarning
    >
      <body>
        <ThemeProvider>
          <I18nProvider>{children}</I18nProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
