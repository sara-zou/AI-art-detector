import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "https://ai-art-detector.onrender.com/api/:path*",
      },
    ] as any;
  },
};

export default nextConfig;