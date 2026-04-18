import type { NextConfig } from "next";

const BACKEND_URL = process.env.BACKEND_URL ?? "http://localhost:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/ws",
        destination: `${BACKEND_URL}/ws`,
      },
    ];
  },
};

export default nextConfig;
