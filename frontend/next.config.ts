import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: process.env.NODE_ENV === "production"
          ? "https://sarazou-ai-art-detector.hf.space/api/:path*"
          : "http://localhost:5001/api/:path*",
      },
    ] as any;
  },
};

export default nextConfig;