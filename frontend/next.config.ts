import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "https://sarazou-ai-art-detector.hf.space/api/:path*",
      },
    ] as any;
  },
};

export default nextConfig;