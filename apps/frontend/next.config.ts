import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Emit a self-contained server bundle so the production Docker image can run
  // `node server.js` without the full node_modules tree.
  output: "standalone",
  // The floating dev-tools button is dev-server-only chrome that a production
  // build never renders, so hiding it costs nothing and keeps a demo clean.
  devIndicators: false,
};

export default nextConfig;
